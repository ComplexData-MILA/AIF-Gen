import logging
import pathlib
import pprint
import random
from dataclasses import asdict
from typing import Optional

import click

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.util.hf import download_from_hf


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--shuffle/--no-shuffle',
    is_flag=True,
    default=True,
    help='Shuffle the order in which samples are previewed.',
)
@click.option(
    '--hf-repo-id',
    type=click.STRING,
    default=None,
    help='If not None, pull the dataset to and from a HuggingFace remote repository with the associated repo-id.',
)
def preview(
    input_data_file: pathlib.Path,
    shuffle: bool,
    hf_repo_id: Optional[str],
) -> None:
    r"""Preview a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    """
    if hf_repo_id is not None:
        input_data_file = download_from_hf(hf_repo_id, input_data_file)

    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(dataset)} samples from: {input_data_file}')

    if len(dataset) == 0:
        logging.error('Dataset is empty!')
        return

    tasks = [
        {
            'AlignmentDataset Index': f'[{i + 1}]/[{len(dataset.datasets)}]',
            'Objective': data.task.objective,
            'Preference': data.task.preference,
        }
        for i, data in enumerate(dataset.datasets)
    ]

    dataset_idx = 0
    while True:
        pprint.pp(tasks[dataset_idx])
        if len(tasks) == 1:
            click.echo('\n> [y]es / [q]uit')
        elif dataset_idx == 0:
            click.echo('\n> [y]es / [n]ext / [q]uit')
        elif dataset_idx == len(tasks) - 1:
            click.echo('\n> [y]es / [p]revious / [q]uit')
        else:
            click.echo('\n> [y]es / [n]ext / [p]revious / [q]uit')

        c = click.getchar()
        click.echo()
        if c.lower() == 'q':
            return
        elif c.lower() == 'n' and dataset_idx < len(tasks) - 1:
            dataset_idx += 1
        elif c.lower() == 'p' and dataset_idx > 0:
            dataset_idx -= 1
        elif c.lower() == 'y':
            splits = [
                dataset.datasets[dataset_idx].train,
                dataset.datasets[dataset_idx].test,
            ]
            if shuffle:
                random.shuffle(splits[0])
                random.shuffle(splits[1])

            if len(splits[0]) == 0 and len(splits[1]) == 0:
                click.echo('No data available!')
                break

            sample_idx, split_idx = 0, 0
            split_names = ['Train', 'Test']
            while True:
                if len(splits[split_idx]) == 0:
                    click.echo(f'No data for {split_names[split_idx]} split!')
                    break

                sample_dict = {
                    f'{split_names[split_idx]} Index': f'{sample_idx + 1}/{len(splits[split_idx])}'
                }
                sample_dict.update(asdict(splits[split_idx][sample_idx]))

                pprint.pp(tasks[dataset_idx])
                pprint.pp(sample_dict)

                other_split = split_names[1 - split_idx]
                if len(splits[split_idx]) == 1:
                    click.echo(f'\n> [s]witch to {other_split} / [b]ack / [q]uit')
                elif sample_idx == 0:
                    click.echo(
                        f'\n> [n]ext / [s]witch to {other_split} / [b]ack / [q]uit'
                    )
                elif sample_idx == len(splits[split_idx]) - 1:
                    click.echo(
                        f'\n> [p]revious / [s]witch to {other_split} / [b]ack / [q]uit'
                    )
                else:
                    click.echo(
                        f'\n> [n]ext / [p]revious / [s]witch to {other_split} / [b]ack / [q]uit'
                    )

                c = click.getchar()
                click.echo()
                if c.lower() == 'q':
                    return
                if c.lower() == 'b':
                    break
                elif c.lower() == 's':
                    split_idx = 1 - split_idx
                    sample_idx = 0
                elif c.lower() == 'n' and sample_idx < len(splits[split_idx]) - 1:
                    sample_idx += 1
                elif c.lower() == 'p' and sample_idx > 0:
                    sample_idx -= 1
