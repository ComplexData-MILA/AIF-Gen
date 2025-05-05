import asyncio
import json
import logging
import pathlib
from typing import Any, Dict, Optional

import click
import openai

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.util.hf import download_from_hf, upload_to_hf
from aif_gen.util.seed import seed_everything
from aif_gen.validation import (
    count_validation,
    entropy_validation,
    llm_embedding_diversity,
    llm_judge_validation,
)


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'output_validation_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--validate-count/--no-validate-count',
    is_flag=True,
    default=True,
    help='Perform basic count validation on the dataset.',
)
@click.option(
    '--validate-entropy/--no-validate-entropy',
    is_flag=True,
    default=True,
    help='Perform entropy validation on the dataset.',
)
@click.option(
    '--validate-llm-judge/--no-validate-llm-judge',
    is_flag=True,
    default=False,
    help='Perform llm judge validation on the dataset.',
)
@click.option(
    '--validate-embedding-diversity/--no-validate-embedding-diversity',
    is_flag=True,
    default=False,
    help='Perform embedding similarity/diversity validation on the dataset.',
)
@click.option(
    '--model',
    type=click.STRING,
    help='vLLM model to use as a judge if doing llm_judge validation',
)
@click.option(
    '--embedding-model',
    type=click.STRING,
    help='vLLM embedding model for computing embedding simiarity',
)
@click.option(
    '--embedding-batch-size',
    type=click.IntRange(min=1),
    default=256,
    help='Number of items to embed in each request.',
)
@click.option(
    '--max_concurrency',
    type=click.IntRange(min=1, max=256, clamp=True),
    help='Max number of concurrent inference requests to send to the vLLM model',
    default=128,
)
@click.option(
    '--max_tokens_judge_response',
    type=click.IntRange(min=1, max=1024, clamp=True),
    help='Limit the max_tokens on the judge response from the vLLM model if doing llm_judge validation.',
    default=128,
)
@click.option(
    '-n',
    '--dry-run',
    is_flag=True,
    default=False,
    help='Ignore the dataset and generate validate a dummy sample to ensure vLLM setup.',
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, pull the dataset to and from a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--random_seed',
    type=int,
    help='Random seed for validation.',
    default=0,
)
def validate(
    input_data_file: pathlib.Path,
    output_validation_file: pathlib.Path,
    validate_count: bool,
    validate_entropy: bool,
    validate_llm_judge: bool,
    validate_embedding_diversity: bool,
    model: str,
    embedding_model: str,
    embedding_batch_size: int,
    max_concurrency: int,
    max_tokens_judge_response: int,
    dry_run: bool,
    hf_repo_id: Optional[str],
    random_seed: int,
) -> None:
    r"""Validate a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_VALIDATION_FILE: Path to the output validation file.
    """
    logging.info(f'Random seed: {random_seed}')
    seed_everything(random_seed)

    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(dataset)} samples from: {input_data_file}')

    results: Dict[str, Any] = {}
    if validate_count:
        logging.info('Performing count validation')
        results['count_validation'] = count_validation(dataset)
        logging.info('Finished count validation')

    if validate_entropy:
        logging.info('Performing entropy validation')
        results['entropy_validation'] = entropy_validation(dataset)
        logging.info('Finished entropy validation')

    if validate_llm_judge:
        logging.info(f'Performing LLM judge validation with model: {model}')

        try:
            client = openai.AsyncOpenAI()
            async_semaphore = asyncio.Semaphore(max_concurrency)
            fut = llm_judge_validation(
                dataset,
                model,
                client,
                async_semaphore,
                max_tokens_judge_response,
                dry_run,
            )
            result = asyncio.get_event_loop().run_until_complete(fut)
        except (openai.OpenAIError, Exception) as e:
            logging.exception(f'Error occurred trying to validate data with vLLM: {e}')
            result = None

        results['llm_judge_validation'] = result
        logging.info('Finished LLM judge validation')

    if validate_embedding_diversity:
        logging.info(
            f'Performing embedding diversity validation with model: {embedding_model}'
        )
        try:
            client = openai.AsyncOpenAI()
            async_semaphore = asyncio.Semaphore(max_concurrency)
            fut = llm_embedding_diversity(
                dataset=dataset,
                model_name=embedding_model,
                client=client,
                batch_size=embedding_batch_size,
                async_semaphore=async_semaphore,
                dry_run=dry_run,
            )
            result = asyncio.get_event_loop().run_until_complete(fut)
        except (openai.OpenAIError, Exception) as e:
            logging.exception(f'Error occurred trying to embed data with vLLM: {e}')
            result = None

        results['llm_embedding_diversity'] = result
        logging.info('Finished embedding similarity')

    if len(results):
        logging.info(f'Writing validation results to: {output_validation_file}')
        with output_validation_file.open('w', encoding='utf-8') as f:
            json.dump(results, f)

        if hf_repo_id is not None:
            upload_to_hf(hf_repo_id, output_validation_file)
    else:
        logging.warning('No validation measure was specified, skipping writedown.')
