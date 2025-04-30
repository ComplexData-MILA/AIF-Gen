import asyncio
import logging
import pathlib
from typing import Optional

import click
import openai

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.generate.service import (
    filter_continual_alignment_dataset_style_normalize,
)
from aif_gen.util.hf import download_from_hf, upload_to_hf
from aif_gen.util.seed import seed_everything


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'output_data_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'model',
    type=click.STRING,
)
@click.option(
    '--max_concurrency',
    type=click.IntRange(min=1, max=256, clamp=True),
    help='Max number of concurrent inference requests to send to the vLLM model',
    default=128,
)
@click.option(
    '--max_tokens',
    type=click.IntRange(min=1, max=65536, clamp=True),
    help='Limit the max_tokens on the chosen-rejected response pair from the vLLM model.',
    default=4096,
)
@click.option(
    '--random_seed',
    type=int,
    help='Random seed for data generation.',
    default=0,
)
@click.option(
    '-n',
    '--dry-run',
    is_flag=True,
    default=False,
    help='Ignore the input config and generate a dummy sample ensuring the model endpoint is setup.',
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, push the generated input_dataset to a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--temperature',
    type=click.FloatRange(min=0.0, max=2.0),
    default=0.99,
    help='Temperature for sampling from the model.',
)
def filter_dataset(
    input_data_file: pathlib.Path,
    output_data_file: pathlib.Path,
    model: str,
    max_concurrency: int,
    max_tokens: int,
    random_seed: int,
    dry_run: bool,
    hf_repo_id: Optional[str],
    temperature: float,
) -> None:
    r"""Filter a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_DATA_FILE: Path to the output dataset.
    MODEL: vLLM-compatible model to use for data filtering.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading input_dataset from: {input_data_file}')
    input_dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(input_dataset)} samples from: {input_data_file}')

    if not len(input_dataset):
        logging.warning('No samples found in dataset, skipping transmutation.')
        return

    logging.info(f'Using model: {model}')
    logging.info(f'Random seed: {random_seed}')
    seed_everything(random_seed)

    output_data_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        client = openai.AsyncOpenAI()
    except (openai.OpenAIError, Exception) as e:
        logging.exception(f'Could not create openAI client: {e}')
        return

    async_semaphore = asyncio.Semaphore(max_concurrency)
    future = filter_continual_alignment_dataset_style_normalize(
        input_dataset,
        model,
        client,
        async_semaphore,
        max_tokens,
        dry_run,
        temperature=temperature,
    )
    dataset = asyncio.get_event_loop().run_until_complete(future)
    if dataset is not None:
        logging.info(f'Writing {len(dataset)} samples to {output_data_file}')
        dataset.to_json(output_data_file)
        logging.info(f'Wrote {len(dataset)} samples to {output_data_file}')

        if hf_repo_id is not None:
            upload_to_hf(repo_id=hf_repo_id, local_path=output_data_file)
