import logging
import pathlib
from typing import Optional

import click

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
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
    'words',
    type=click.STRING,
)
@click.option(
    '--random_seed',
    type=int,
    help='Random seed for data generation.',
    default=0,
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, push the generated input_dataset to a HuggingFace remote repository with the associated repo-id.',
)
def clean_dataset(
    input_data_file: pathlib.Path,
    output_data_file: pathlib.Path,
    words: str,
    random_seed: int,
    hf_repo_id: Optional[str],
) -> None:
    r"""Clean a ContinualAlignmentDataset given a space-separated string of words.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_DATA_FILE: Path to the output dataset.
    WORDS: Space-separated string of words to clean the dataset.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading input_dataset from: {input_data_file}')
    input_dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(input_dataset)} samples from: {input_data_file}')

    if not len(input_dataset):
        logging.warning('No samples found in dataset, skipping clean up.')
        return

    logging.info(f'Using words: {words}')
    logging.info(f'Random seed: {random_seed}')
    seed_everything(random_seed)

    output_data_file.parent.mkdir(parents=True, exist_ok=True)

    words_list = words.split(' ')
    if len(words_list) == 0:
        logging.warning('No words found in words string, skipping clean up.')
        return

    # clean up each data point in the dataset
    for dataset in input_dataset.datasets:
        for sample in dataset.samples:
            for word in words_list:
                sample.prompt = sample.prompt.replace(word, '')
                sample.chosen = sample.chosen.replace(word, '')
                sample.rejected = sample.rejected.replace(word, '')

    logging.info(f'Finished cleaning dataset.')

    logging.info(f'Writing {len(dataset)} samples to {output_data_file}')
    input_dataset.to_json(output_data_file)
    logging.info(f'Wrote {len(dataset)} samples to {output_data_file}')

    if hf_repo_id is not None:
        upload_to_hf(repo_id=hf_repo_id, local_path=output_data_file)
