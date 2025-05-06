import pathlib

import click

from aif_gen.cli.commands.generate import generate
from aif_gen.cli.commands.merge import merge
from aif_gen.cli.commands.preview import preview
from aif_gen.cli.commands.sample import sample
from aif_gen.cli.commands.transform import transform
from aif_gen.cli.commands.validate import validate
from aif_gen.util.logging import setup_basic_logging


class RichGroup(click.Group):
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        s = """
          / _ | /  _/ __/ / ___/ __/ |/ /
         / __ |_/ // _/  / (_ / _//    /
        /_/ |_/___/_/    \___/___/_/|_/"""

        s += '\n\nA tool for generating synthetic continual RLHF datasets.\n\n'

        formatter.write(s)
        super().format_help(ctx, formatter)


@click.group(cls=RichGroup, context_settings={'show_default': True})
@click.option(
    '--log_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help='Optional log file to use.',
    default=f'aif_gen.log',
)
def cli(log_file: pathlib.Path) -> None:
    setup_basic_logging(log_file)


cli.add_command(generate)
cli.add_command(validate)
cli.add_command(preview)
cli.add_command(merge)
cli.add_command(transform)
cli.add_command(sample)

if __name__ == '__main__':
    cli()
