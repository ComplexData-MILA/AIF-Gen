import logging
import pathlib
from typing import Optional

import click

import aif_gen.transforms.functional as F
from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.util.hf import download_from_hf, upload_to_hf


@click.group()
def transform() -> None:
    r"""Transform a ContinualAlignmentDataset."""


@transform.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'output_data_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, pull and push the transformed dataset to and from a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--p',
    type=click.FloatRange(min=0, max=1),
    default=1,
    help="Probability with which to swap each 'chosen' and 'rejected' in the dataset",
)
def preference_swap(
    input_data_file: pathlib.Path,
    output_data_file: pathlib.Path,
    p: float,
    hf_repo_id: Optional[str],
) -> None:
    r"""Swap the 'chosen' and 'rejected' responses for each sample in the dataset with probability.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_DATA_FILE: Path to the output (transformed) dataset.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(dataset)} samples from: {input_data_file}')

    logging.info(f'Applying preference swap transform with p={p}')
    transformed_dataset = F.preference_swap_transform(dataset, swap_probability=p)
    logging.info(f'Transformed dataset.')

    logging.info(f'Writing dataset to: {output_data_file}')
    transformed_dataset.to_json(output_data_file)
    logging.info(f'Wrote {len(transformed_dataset)} samples from: {output_data_file}')

    if hf_repo_id is not None:
        upload_to_hf(hf_repo_id, output_data_file)
