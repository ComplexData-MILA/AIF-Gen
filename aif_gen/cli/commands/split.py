import logging
import pathlib
from typing import Optional

import click

import aif_gen.dataset.split.functional as F
from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.util.hf import download_from_hf, upload_to_hf
from aif_gen.util.path import get_run_id
from aif_gen.util.seed import seed_everything


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, pull the dataset to and from a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--hf-repo-id-out',
    type=click.STRING,
    default=None,
    help='If not None, push the dataset to a HuggingFace remote repository with the associated repo-id.',
)
@click.option(
    '--output_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help='Path to write the generated dataset.',
    default=lambda: f'data/{get_run_id(name=click.get_current_context().params["input_data_file"].stem)}/data.json',
)
@click.option(
    '--test_sample_ratio',
    type=click.FloatRange(min=0.0, max=1.0, clamp=True),
    help='Ratio of samples to use for testing in each static task of the dataset.',
    default=0.15,
)
@click.option(
    '--random_seed',
    type=int,
    help='Random seed for test data selection.',
    default=0,
)
def split(
    input_data_file: pathlib.Path,
    hf_repo_id: Optional[str],
    hf_repo_id_out: Optional[str],
    output_file: pathlib.Path,
    test_sample_ratio: float,
    random_seed: int,
) -> None:
    r"""Split a ContinualAlignmentDataset into train and test datasets.

    INPUT_DATA_FILE: Path to the input dataset.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {dataset.num_samples} samples from: {input_data_file}')

    if len(dataset) == 0:
        logging.error('Dataset is empty!')
        return

    seed_everything(random_seed)
    logging.info(f'Splitting dataset with test_sample_ratio={test_sample_ratio}')
    dataset = F.split(dataset, test_ratio=test_sample_ratio)
    logging.info(f'Writing dataset to: {output_file}')
    dataset.to_json(output_file)
    logging.info(f'Wrote {dataset.num_samples} samples to: {output_file}')

    if hf_repo_id_out is not None:
        upload_to_hf(hf_repo_id_out, output_file)
        logging.info(f'Uploaded dataset to HuggingFace repo: {hf_repo_id_out}')
    else:
        logging.info(f'No HuggingFace repo specified for upload.')

    return
