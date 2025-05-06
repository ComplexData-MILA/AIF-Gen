from pydantic.dataclasses import dataclass


@dataclass
class AlignmentDatasetSample:
    r"""Container for a single Alignment Dataset Sample.

    This representation is faithful to the "TRL Preference Format with explicit prompt".
    See: https://huggingface.co/docs/trl/en/dataset_formats.

    Args:
        prompt (str): The prompt associated with the sample.
        chosen (str): The winning response associated with the sample.
        rejected (str): The losing response associated with the sample.
    """

    prompt: str
    chosen: str
    rejected: str

    def __str__(self) -> str:
        return (
            f'Prompt: {self.prompt}, Chosen: {self.chosen}, Rejected: {self.rejected}'
        )
