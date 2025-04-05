from typing import List

import click

from aif_gen.dataset.alignment_dataset import AlignmentDataset
from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)


@click.command()
def merge() -> None:
    r"""Merge a set of ContinualAlignmentDatasets."""
    merged: List[AlignmentDataset] = []
    total_samples = 0
    while True:
        click.echo(
            f'{len(merged)} Datasets Buffered ({total_samples} samples)\t\t[a]dd / [m]erge / q[uit]'
        )
        c = click.getchar()
        click.echo()
        if c.lower() == 'q':
            return
        elif c.lower() == 'm':
            break
        elif c.lower() == 'a':
            path = click.prompt('> Path to dataset')
            try:
                dataset = ContinualAlignmentDataset.from_json(path)
            except Exception as e:
                click.secho(f'Failed to read dataset from {path}: {e}', fg='red')
                continue
            click.secho(
                f'Read ContinualAlignmentDataset with {len(dataset.datasets)} constituents'
                f' and {dataset.num_samples} samples',
                fg='green',
            )
            merged.extend(dataset.datasets)
            total_samples += len(dataset)

    if not len(merged):
        click.secho(f'No datasets in buffer, skipping writedown', fg='red')
        return

    click.echo(f'Merging {len(merged)} datasets')
    dataset = ContinualAlignmentDataset(merged)
    path = click.prompt('> Path to save merged dataset')
    click.echo(f'Writing dataset to: {path}')
    dataset.to_json(path)
    click.secho(f'Wrote {len(dataset)} samples to: {path}', fg='green')
