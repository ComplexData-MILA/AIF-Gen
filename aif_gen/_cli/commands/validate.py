import json
import logging
import os
import pathlib
from typing import Any, Dict

import click

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation import (
    count_validation,
    diversity_validation,
    entropy_validation,
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
    '--validate-diversity/--no-validate-diversity',
    is_flag=True,
    default=True,
    help='Perform diversity validation on the dataset.',
)
@click.option(
    '--validate-llm-judge/--no-validate-llm-judge',
    is_flag=True,
    default=True,
    help='Perform llm judge validation on the dataset.',
)
@click.option(
    '--bleu_batch_size',
    type=click.IntRange(min=1, clamp=True),
    help=(
        'Number BLEU reference examples to process at one time. '
        'Reduce this value if JAX reports OOM'
    ),
    default=256,
)
def validate(
    input_data_file: pathlib.Path,
    output_validation_file: pathlib.Path,
    validate_count: bool,
    validate_entropy: bool,
    validate_diversity: bool,
    validate_llm_judge: bool,
    bleu_batch_size: int,
) -> None:
    r"""Validate a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_VALIDATION_FILE: Path to the output validation file.
    """
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

    if validate_diversity:
        logging.info('Performing diversity validation')
        results['diversity_validation'] = diversity_validation(dataset, bleu_batch_size)
        logging.info('Finished diversity validation')

    if validate_llm_judge:
        logging.info('Performing LLM judge validation')
        results['llm_judge_validation'] = llm_judge_validation(dataset)
        logging.info('Finished LLM judge validation')

    if len(results):
        logging.info(f'Writing validation results to: {output_validation_file}')
        with output_validation_file.open('w', encoding='utf-8') as f:
            json.dump(results, f)
    else:
        logging.warning('No validation measure was specified, skipping writedown.')
