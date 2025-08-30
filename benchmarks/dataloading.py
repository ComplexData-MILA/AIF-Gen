from pathlib import Path
from typing import Any, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

from aif_gen.dataset import ContinualAlignmentDataset


def process_datasets_for_alignment(
    datasets_list: list[dict[str, Dataset]],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tools: Optional[list] = None,
) -> list[dict[str, Dataset]]:
    """Process each dataset in the list to prepare it for alignment training.

    Args:
        datasets_list: List of dictionaries containing datasets (typically with 'train' and 'test' keys)
        tokenizer: Tokenizer to use for applying chat templates
        tools: Tools to use for chat template application

    Returns:
        Processed list of dictionaries with datasets ready for alignment training
    """
    processed_datasets = []

    for dataset_dict in datasets_list:
        processed_dict = {}

        for key, dataset in dataset_dict.items():
            # Apply the processing steps
            # Step 1: Extract prompts
            processed_dataset = dataset.map(
                maybe_extract_prompt, desc=f'Extracting prompts in {key} dataset'
            )

            # Step 2: Apply chat template if tokenizer is provided
            if tokenizer is not None:
                processed_dataset = processed_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={'tokenizer': tokenizer, 'tools': tools},
                    desc=f'Applying chat template to {key} dataset',
                )

            processed_dict[key] = processed_dataset

        processed_datasets.append(processed_dict)

    return processed_datasets


# CONVENTION: Only return explicit preferences dataset
def _init_mock_dataset(
    dataset_name: Union[str, ContinualAlignmentDataset, Path],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tools: Optional[list] = None,
) -> list[dict[str, Dataset]]:
    if dataset_name == 'debug':
        datasets: list[dict[str, Any]] = get_debug_datasets()
    elif dataset_name == 'ultrafeedback2anthropic':
        datasets = get_ultrafeedback2anthropic_datasets()
    elif dataset_name == 'ultrafeedback2anthropic_reduced':
        datasets = get_ultrafeedback2anthropic_datasets_reduced()
    elif dataset_name == 'cppo-reward':
        datasets = get_CPPO_reward_dataset()
    elif dataset_name == 'cppo-rl':
        datasets = get_CPPO_rl_dataset()
    else:
        raise ValueError(f'Unknown mock dataset: {dataset_name}')

    # Process datasets to prepare them for alignment training
    return process_datasets_for_alignment(datasets, tokenizer=tokenizer, tools=tools)


def init_continual_dataset(
    dataset: Union[str, ContinualAlignmentDataset, Path],
    mock: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tools: Optional[list] = None,
) -> list[dict[str, Dataset]]:
    """Initialize a continual dataset from a given dataset name or path or a ContinualAlignmentDataset Object."""
    if mock:
        return _init_mock_dataset(dataset, tokenizer=tokenizer, tools=tools)
    if not isinstance(dataset, ContinualAlignmentDataset):
        try:
            data = ContinualAlignmentDataset.from_json(dataset)
        except OSError:  # need to try downloading from hub
            try:
                # print(f'Downloading {json_name} from Hugging Face Hub...')
                local_path = hf_hub_download(
                    repo_id=f'LifelongAlignment/{dataset}',
                    filename='data.json',
                    repo_type='dataset',
                )
                data = ContinualAlignmentDataset.from_json(local_path)
            except Exception as e:
                raise ValueError(f'Error loading dataset: {e}')
    return data.to_hf_compatible()


def get_debug_datasets() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100)),
        },
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100, 200)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100, 200)),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train').select(
                range(100)
            ),
            'test': load_dataset('Anthropic/hh-rlhf', split='test').select(range(100)),
        },
    ]
    return datasets


def get_ultrafeedback2anthropic_datasets_reduced() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(35200)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(1000)),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train').select(
                range(35200)
            ),
            'test': load_dataset('Anthropic/hh-rlhf', split='test').select(range(1000)),
        },
    ]
    return datasets


def get_ultrafeedback2anthropic_datasets() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset('trl-lib/ultrafeedback_binarized', split='train'),
            'test': load_dataset('trl-lib/ultrafeedback_binarized', split='test'),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train'),
            'test': load_dataset('Anthropic/hh-rlhf', split='test'),
        },
    ]

    return datasets


def save_CPPO_datasets_to_hub() -> None:
    """Process CPPO datasets and save them to Hugging Face Hub."""

    def _preprocess_CPPO_datasets() -> (
        tuple[list[dict[str, Dataset]], list[dict[str, Dataset]]]  # noqa: E501
    ):
        """Datasets used for task incremental learning in CPPO: https://openreview.net/forum?id=86zAUE80pP.

        Dataset: https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons?row=0

        reward model training:
        NOTE: the reward model is trained continually on the data below.
            task 1 - r/relationships:
                - Train: 52243
                - valid: 0
                - Test: 45148
            task 2 r/others (everything but r/relationships):
                - Train: 40291
                - valid: 0
                - Test: 38481

        RL training (using prompts from the same dataset):
            task 1 - r/relationships:
                - Train: 52243
                - valid: 3462
                - Test: 45148
            task 2 - r/others:
                - Train: 40291
                - valid: 2985
                - Test: 38481

        Returns:
            Tuple of datasets for reward model training and RL training
        """
        # Load datasets once
        print('Loading CarperAI comparison dataset...')
        comparison_train = load_dataset(
            'CarperAI/openai_summarize_comparisons', split='train'
        )
        comparison_test = load_dataset(
            'CarperAI/openai_summarize_comparisons', split='test'
        )

        # For CarperAI dataset, extract subreddit from prompt
        def extract_subreddit(example: dict) -> dict:
            prompt = example.get('prompt', '')
            if 'r/relationships' in prompt.lower():
                example['subreddit'] = 'relationships'
            else:
                example['subreddit'] = 'others'
            return example

        print('Extracting subreddit information...')
        comparison_train = comparison_train.map(extract_subreddit)
        comparison_test = comparison_test.map(extract_subreddit)

        # Filter datasets by subreddit
        def is_relationships(example: dict) -> bool:
            return example.get('subreddit') == 'relationships'

        def is_others(example: dict) -> bool:
            return example.get('subreddit') == 'others'

        # Filter and sample datasets for tasks
        print('Preparing task datasets...')
        train_relationships = comparison_train.filter(is_relationships)
        train_others = comparison_train.filter(is_others)
        test_relationships = comparison_test.filter(is_relationships)
        test_others = comparison_test.filter(is_others)

        # Sample according to specified sizes for reward model
        rm_train_task1 = train_relationships.select(
            range(min(52243, len(train_relationships)))
        )
        rm_test_task1 = test_relationships.select(
            range(min(45148, len(test_relationships)))
        )
        rm_train_task2 = train_others.select(range(min(40291, len(train_others))))
        rm_test_task2 = test_others.select(range(min(38481, len(test_others))))

        # Process reward model datasets
        def process_for_reward_model(dataset: list[dict]) -> Dataset:
            processed_samples = []
            for sample in tqdm(dataset):
                prompt = sample['prompt']
                chosen = sample['chosen']
                rejected = sample['rejected']

                # Quality filtering
                if chosen == rejected:
                    continue
                if len(chosen.split()) < 5 or len(rejected.split()) < 5:
                    continue

                processed_samples.append(
                    {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
                )

            # print the number of dropped samples
            print(f'dropped samples: {len(dataset) - len(processed_samples)}')

            return Dataset.from_list(processed_samples)

        print('Processing reward model datasets...')
        reward_model_datasets = [
            {
                'train': process_for_reward_model(rm_train_task1),
                'test': process_for_reward_model(rm_test_task1),
            },
            {
                'train': process_for_reward_model(rm_train_task2),
                'test': process_for_reward_model(rm_test_task2),
            },
        ]

        # Process RL datasets (just use prompts from the same data)
        def process_for_rl(dataset: list[dict]) -> Dataset:
            processed_samples = []
            for sample in tqdm(dataset):
                prompt = sample['prompt']
                processed_samples.append(
                    {
                        'prompt': prompt,
                        # For evaluation/reference only
                        'chosen': sample['chosen'],
                        'rejected': sample['rejected'],
                    }
                )

            return Dataset.from_list(processed_samples)

        print('Processing RL datasets...')
        # For RL, include validation sets by splitting training data - for our training, we don't use validation sets
        rl_train_task1 = rm_train_task1.select(
            range(min(52243 - 3462, len(rm_train_task1)))
        )
        rl_test_task1 = rm_test_task1

        rl_train_task2 = rm_train_task2.select(
            range(min(40291 - 2985, len(rm_train_task2)))
        )
        rl_test_task2 = rm_test_task2

        rl_datasets = [
            {
                'train': process_for_rl(rl_train_task1),
                'test': process_for_rl(rl_test_task1),
            },
            {
                'train': process_for_rl(rl_train_task2),
                'test': process_for_rl(rl_test_task2),
            },
        ]

        return reward_model_datasets, rl_datasets

    reward_model_datasets, rl_datasets = _preprocess_CPPO_datasets()

    # Convert to DatasetDict for easier saving
    reward_model_dataset_dicts = [
        DatasetDict(
            {
                'train': reward_model_datasets[0]['train'],
                'test': reward_model_datasets[0]['test'],
            }
        ),
        DatasetDict(
            {
                'train': reward_model_datasets[1]['train'],
                'test': reward_model_datasets[1]['test'],
            }
        ),
    ]

    rl_dataset_dicts = [
        DatasetDict({'train': rl_datasets[0]['train'], 'test': rl_datasets[0]['test']}),
        DatasetDict({'train': rl_datasets[1]['train'], 'test': rl_datasets[1]['test']}),
    ]

    # Save reward model datasets to Hugging Face Hub
    print('Saving reward model datasets to Hugging Face Hub...')
    reward_model_dataset_dicts[0].push_to_hub(
        'Shahradmz/cppo_continual_dataset_reward_relationships'
    )
    reward_model_dataset_dicts[1].push_to_hub(
        'Shahradmz/cppo_continual_dataset_reward_others'
    )

    # Save RL datasets to Hugging Face Hub
    print('Saving RL datasets to Hugging Face Hub...')
    rl_dataset_dicts[0].push_to_hub('Shahradmz/cppo_continual_dataset_rl_relationships')
    rl_dataset_dicts[1].push_to_hub('Shahradmz/cppo_continual_dataset_rl_others')

    print('Datasets saved successfully.')


def get_CPPO_reward_dataset() -> list[dict[str, Dataset]]:
    """Get the CPPO reward model datasets for alignment training.

    Returns:
        List of dictionaries containing the train and test datasets for each task
    """
    datasets = [
        {
            'train': load_dataset(
                'Shahradmz/cppo_continual_dataset_reward_relationships', split='train'
            ),
            'test': load_dataset(
                'Shahradmz/cppo_continual_dataset_reward_relationships', split='test'
            ),
        },
        {
            'train': load_dataset(
                'Shahradmz/cppo_continual_dataset_reward_others', split='train'
            ),
            'test': load_dataset(
                'Shahradmz/cppo_continual_dataset_reward_others', split='test'
            ),
        },
    ]
    return datasets


def get_CPPO_rl_dataset() -> list[dict[str, Dataset]]:
    """Get the CPPO RL datasets for training reinforcement learning models.

    Returns:
        List of dictionaries containing the train and test datasets for each task
    """
    datasets = [
        {
            'train': load_dataset(
                'Shahradmz/cppo_continual_dataset_rl_relationships', split='train'
            ),
            'test': load_dataset(
                'Shahradmz/cppo_continual_dataset_rl_relationships', split='test'
            ),
        },
        {
            'train': load_dataset(
                'Shahradmz/cppo_continual_dataset_rl_others', split='train'
            ),
            'test': load_dataset(
                'Shahradmz/cppo_continual_dataset_rl_others', split='test'
            ),
        },
    ]
    return datasets
