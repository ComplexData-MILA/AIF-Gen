import logging
import pathlib
import random
from typing import Optional

import click
from tqdm import tqdm

from aif_gen.dataset.alignment_dataset import AlignmentDataset
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
@click.argument(
    'keep_ratio_train',
    type=click.FloatRange(min=0.0, max=1.0, clamp=True),
)
@click.argument(
    'keep_ratio_test',
    type=click.FloatRange(min=0.0, max=1.0, clamp=True),
)
@click.option(
    '--keep_amount_train',
    type=int,
    help='Amount of samples to keep in the dataset. If not None, overrides the keep_ratio_train and keep_ratio_test options.',
    default=None,
)
@click.option(
    '--keep_amount_test',
    type=int,
    help='Amount of samples to keep in the dataset. If not None, overrides the keep_ratio_train and keep_ratio_test options.',
    default=None,
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
    '--random_seed',
    type=int,
    help='Random seed for test data selection.',
    default=0,
)
def sample(
    input_data_file: pathlib.Path,
    keep_ratio_train: float,
    keep_ratio_test: float,
    keep_amount_train: Optional[int],
    keep_amount_test: Optional[int],
    hf_repo_id: Optional[str],
    hf_repo_id_out: Optional[str],
    output_file: pathlib.Path,
    random_seed: int,
) -> None:
    r"""Downsample a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    KEEP_RATIO_TRAIN: Ratio of samples to keep in the train dataset.
    KEEP_RATIO_TEST: Ratio of samples to keep in the test dataset.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {dataset.num_samples} samples from: {input_data_file}')

    if len(dataset) == 0:
        logging.error('Dataset is empty!')
        return

    logging.info(f'Original dataset has {dataset.num_samples} samples.')
    logging.info(f'Original dataset has {dataset.num_datasets} tasks.')
    logging.info(f'Starting sampling with seed {random_seed} for each task.')
    seed_everything(random_seed)

    for i, data in tqdm(enumerate(dataset.datasets), desc='Processing datasets'):
        train, test = data.train, data.test

        train_size = int(keep_ratio_train * len(train))
        if keep_amount_train is not None:
            train_size = keep_amount_train

        test_size = int(keep_ratio_test * len(test))
        if keep_amount_test is not None:
            test_size = keep_amount_test

        train_size = min(train_size, len(train))
        test_size = min(test_size, len(test))

        new_train = random.sample(train, train_size)
        new_test = random.sample(test, test_size)
        dataset.datasets[i] = AlignmentDataset(
            task=data.task, samples=new_train + new_test, train_frac=data.train_frac
        )

    logging.info(f'Writing dataset to: {output_file}')
    dataset.to_json(output_file)
    logging.info(f'Wrote {dataset.num_samples} samples to: {output_file}')

    if hf_repo_id_out is not None:
        upload_to_hf(hf_repo_id_out, output_file)
        logging.info(f'Uploaded dataset to HuggingFace repo: {hf_repo_id_out}')
