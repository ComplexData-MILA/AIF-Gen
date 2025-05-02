import datetime
import subprocess


def get_run_id(name: str = 'aif_data') -> str:
    r"""Create a unique identifier associated with an experiment.

    Args:
        name (str): Prefix string associated with the run identifier.

    Returns:
        A run id of the form: {name}/{time}_{git_hash}
    """

    def get_git_revision_short_hash() -> str:
        return (
            subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            .decode('ascii')
            .strip()
        )

    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hash = get_git_revision_short_hash()
    return f'{name}/{time}_{hash}'
