import logging
import pathlib

from huggingface_hub import HfApi, hf_hub_download


def upload_to_hf(repo_id: str, local_path: str | pathlib.Path) -> None:
    r"""Upload a local dataset to a remote HuggingFace repository.

    Args:
        repo_id (str): The name of the HuggingFace dataset repository.
        local_path (Union[str, pathlib.Path]): Local path to upload, either a single file, or directory.

    Note: Assumes the client is authenticated.
    """
    local_path = pathlib.Path(local_path)

    api = HfApi()
    logging.info(f'Creating HuggingFace repo: {repo_id} if it does not exist')
    api.create_repo(repo_id, exist_ok=True, repo_type='dataset')
    logging.info(f'HuggingFace repo: {repo_id} exists')

    if local_path.is_file():
        logging.info(f'Uploading local file: {local_path} to {repo_id}')
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=str(local_path.name),
            repo_id=repo_id,
            repo_type='dataset',
        )
        logging.info(f'Uploaded local file: {local_path} to {repo_id}')
    else:
        logging.info(f'Uploading local folder: {local_path} to {repo_id}')
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type='dataset',
            allow_patterns='*.json',
        )
        logging.info(f'Uploaded local folder: {local_path} to {repo_id}')


def download_from_hf(repo_id: str, filename: str | pathlib.Path) -> pathlib.Path:
    r"""Download a remote HuggingFace dataset to the local file system.

    Args:
        repo_id (str): The name of the HuggingFace dataset repository.
        filename(Union[str, pathlib.Path]): Remote path to download.

    Returns:
        Absolute path on the local filesystem where the data was downloaded.

    Note: Assumes the client is authenticated.
    """
    if isinstance(filename, pathlib.Path):
        filename = str(filename)

    logging.info(f'Downloading {filename} from {repo_id}')
    local_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type='dataset'
    )
    logging.info(f'Downloaded {filename} from {repo_id} to {local_path}')
    return pathlib.Path(local_path)
