import asyncio
import copy
import json
import logging
import pathlib
from typing import Optional

import click
import openai
import yaml  # type: ignore

from aif_gen.generate.engine import generate_continual_dataset
from aif_gen.util.hf import upload_to_hf
from aif_gen.util.path import get_run_id
from aif_gen.util.seed import seed_everything


@click.command(context_settings={'show_default': True})
@click.argument(
    'data_config_name',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'model',
    type=click.STRING,
)
@click.option(
    '--output_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help='Path to write the generated dataset.',
    default=lambda: f'data/{get_run_id(name=click.get_current_context().params["data_config_name"].stem)}/data.json',
)
@click.option(
    '--max_concurrency',
    type=click.IntRange(min=1, max=256, clamp=True),
    help='Max number of concurrent inference requests to send to the vLLM model',
    default=128,
)
@click.option(
    '--max_tokens_prompt_response',
    type=click.IntRange(min=1, max=32768, clamp=True),
    help='Limit the max_tokens on the prompt response from the vLLM model.',
    default=1024,
)
@click.option(
    '--max_tokens_chosen_rejected_response',
    type=click.IntRange(min=1, max=65536, clamp=True),
    help='Limit the max_tokens on the chosen/rejected response pair from the vLLM model.',
    default=2048,
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
    help='If not None, push the generated dataset to a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--include-preference-axes',
    is_flag=True,
    default=False,
    help='Include preference axes in the generated dataset.',
)
@click.option(
    '--temperature',
    type=click.FloatRange(min=0.0, max=2.0, clamp=True),
    default=0.99,
    help='Temperature for sampling from the model.',
)
def generate(
    data_config_name: pathlib.Path,
    model: str,
    output_file: pathlib.Path,
    max_concurrency: int,
    max_tokens_prompt_response: int,
    max_tokens_chosen_rejected_response: int,
    random_seed: int,
    dry_run: bool,
    hf_repo_id: Optional[str],
    include_preference_axes: bool,
    temperature: float,
) -> None:
    r"""Generate a new ContinualAlignmentDataset.

    DATA_CONFIG_NAME: Path to the dataset configuration file to use for dataset generation.
    MODEL: vLLM-compatible model to use for data generation.
    """
    logging.info(f'Using data configuration file: {data_config_name}')
    logging.info(f'Using model: {model}')
    logging.info(f'Random seed: {random_seed}')
    seed_everything(random_seed)

    data_config = yaml.safe_load(data_config_name.read_text())
    logging.debug(f'Configuration: {data_config}')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        config = copy.deepcopy(data_config)
        config['model'] = model
        config['max_concurrency'] = max_concurrency
        with open(output_file.parent / 'config.json', 'w') as f:
            json.dump(config, f)

    try:
        client = openai.AsyncOpenAI()
    except (openai.OpenAIError, Exception) as e:
        logging.exception(f'Could not create openAI client: {e}')
        return

    async_semaphore = asyncio.Semaphore(max_concurrency)
    future = generate_continual_dataset(
        data_config,
        model,
        client,
        async_semaphore,
        max_tokens_prompt_response,
        max_tokens_chosen_rejected_response,
        dry_run,
        include_preference_axes=include_preference_axes,
        temperature=temperature,
    )
    dataset = asyncio.get_event_loop().run_until_complete(future)
    if dataset is not None:
        logging.info(f'Writing {len(dataset)} samples to {output_file}')
        dataset.to_json(output_file)
        logging.info(f'Wrote {len(dataset)} samples to {output_file}')

        if hf_repo_id is not None:
            upload_to_hf(repo_id=hf_repo_id, local_path=output_file)
