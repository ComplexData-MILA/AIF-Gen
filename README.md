<a id="readme-top"></a>

![image](./docs/img/logo.svg)

<div align="center">
<h3 style="font-size: 22px">Generating Synthetic Continual RLHF Data at Scale</h3>
<a href="https://aif-gen.readthedocs.io/en/latest"/><strong style="font-size: 18px;">Read Our Docs»</strong></a>
<a href="https://github.com/ComplexData-MILA/AIF-Gen"/><strong style="font-size: 18px;">Read Our Paper»</strong></a>
<br/>
<br/>

[![GitHub Repo stars](https://img.shields.io/github/stars/ComplexData-MILA/AIF-Gen)](https://github.com/ComplexData-MILA/AIF-Gen/stargazers)
[![Unit Tests](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/testing.yml/badge.svg)](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/testing.yml)
[![Linting](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/ruff.yml/badge.svg)](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/ruff.yml)

</div>

## About The Project

AIF-Gen is a platform for generating synthetic RLHF datasets for lifelong reinforcement learning on LLMs.

Our main goal is to facilitate preference data generation at scale via [RL from AI feedback](https://arxiv.org/abs/2309.00267). AIF-Gen natively supports evolving preferences making it especially useful for studying non-stationary domains such as tutoring. Think of it like [Procgen](https://github.com/openai/procgen), but for RLHF.

> \[!NOTE\]
> AIF-Gen is still alpha software, and may introduce breaking changes.

### Library Highlights

- :zap: Asynchronous LLM batch inference powered by [vLLM](https://github.com/vllm-project/vllm)
- :wrench: Modular prompt templates and fully customizable preference specification
- :file_cabinet: LLM response cache to avoid redundant API requests
- :white_check_mark: Validation metrics to judge synthetic data quality
- 🤗 Direct integration with HuggingFace for robust dataset management

### Architecture Overview

![image](./docs/img/architecture-dark.svg#gh-dark-mode-only)
![image](./docs/img/architecture-light.svg#gh-light-mode-only)

## Quick Tour for New Users

AIF-Gen is intended to be primarily used as a command line tool:

```console
foo@bar:~$ aif --help

          / _ | /  _/ __/ / ___/ __/ |/ /
         / __ |_/ // _/  / (_ / _//    /
        /_/ |_/___/_/    \___/___/_/|_/

A tool for generating synthetic continual RLHF datasets.

Usage: aif [OPTIONS] COMMAND [ARGS]...

Options:
  --log_file FILE  Optional log file to use.  [default: aif_gen.log]
  --help           Show this message and exit.

Commands:
  generate   Generate a new ContinualAlignmentDataset.
  merge      Merge a set of ContinualAlignmentDatasets.
  preview    Preview a ContinualAlignmentDataset.
  sample     Downsample a ContinualAlignmentDataset.
  transform  Transform a ContinualAlignmentDataset.
  validate   Validate a ContinualAlignmentDataset.
```

For advanced usage, refer to [our docs](https://aif-gen.readthedocs.io/).

### Generating Data

In this example, we highlight the ease of generating synthetic data with AIF-Gen.

#### Pre-requisites

First, ensure you have installed AIF-Gen (see [installation](#installation)).
For this example, we'll generating data using [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf).
The chat template we are using is found [here](./chat_templates/olmo-chat-template.jinja).

We'll need to serve our model on an inference server with vLLM. The following will do the trick:

```sh
# Install vLLM (only needs to be done once)
uv tool install vllm

# Serve the model locally
uvx --with setuptools serve allenai/OLMo-1B-hf --dtype auto --api-key MY_KEY --chat-template chat_templates/omlo-chat-template.jinja
```

Some things to keep in mind:

- We use the api-key `MY_KEY`, but anything works here
- This starts an inference server listening on `localhost:8000`

#### Export env variables

Now that the inference server is up, we'll need to export a few environment variables so that AIF-Gen knows where to direct requests.

```sh
export OPENAI_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=MY_KEY

# Optionally, set the following to cache OpenAI requests in Elasticsearch.
export ELASTIC_SEARCH_HOST="..."
export ELASTIC_SEARCH_API_KEY="..."
```

#### Create a Dataset Configuration

We are now ready to specify our preference data configuration. We'll create the following yaml file in `config/philosophy_qna.yaml`.

```yaml
---
task_specs:
  # First dataset: 5 samples of Philosophy QNA with ELI5 preference
  - num_samples: 5
    alignment_task:
      objective: 'Ask an interesting philosophy question'
      preference: 'Explain the answer at a level that could be understood by a 5 year old'
      domain:
        philosophy:
          seed_words: # Some interesting words we want inject into our prompts
            - consciousness
            - time
            - metaphysics

  # Second dataset, 5 samples of Philosophy QNA with expert preference
  - num_samples: 5
    alignment_task:
      objective: 'Ask an interesting philosophy question'
      preference: 'Explain the answer at an expert level. Draw from technical literature.'
      domain:
        philosophy:
          seed_words: # Change up some seed words for variety
            - determinism
            - universe
            - meaning
```

This will produce a final dataset with 10 samples in [TRL preference format with explicit prompts](https://huggingface.co/docs/trl/en/dataset_formats).
The first 5 responses follow the *ELI5* preference, while the last 5 should be more technical.

#### Generate some data

It's advisable to do a dry run first to ensure everything is setup correctly:

```sh
aif generate config/philosophy_qna.yaml allenai/OLMo-1B-hf --dry-run
```

If everything worked, you should see: `Dry run was a success`. We can now generate the data:

```sh
aif generate config/philosophy_qna.yaml allenai/OLMo-1B-hf
```

For options such as choosing output directory, changing model temperature, increasing concurrency limits, and uploading directly to hugging face, check our docs or issue `aif generate --help`.

> \[!TIP\]
> Refer to our [our docs](https://aif-gen.readthedocs.io/) for information and example usage for the other commands.

## Installation

The current recommended way to install AIF-Gen is from source.

#### Using [uv](https://docs.astral.sh/uv/) (recommended)

```sh
# Create and activate your venv
uv venv my_venv --python 3.10 && source my_venv/bin/activate

# Install the wheels into the venv
uv pip install git+https://github.com/ComplexData-MILA/AIF-Gen.git

# Test the install
aif
```

#### Using [pip](https://pip.pypa.io/en/stable/installation/)

```sh
# Create and activate your venv
python3.10 -m venv my_venv && source my_venv/bin/activate

# Install the wheels into the venv
pip install git+https://github.com/ComplexData-MILA/AIF-Gen.git

# Test the install
aif
```

## Documentation

Documentation along with a quick start guide can be found on the [docs website](https://aif-gen.readthedocs.io/).

## Citation

Please cite [our paper](https://github.com/ComplexData-MILA/AIF-Gen) if your use this code in your own work:

```
@article{TODO,
  title   = "TODO",
  author  = "TODO"
  journal = "TODO",
  url     = "TODO"
  year    = "2025",
}
```

## Contributing

If you notice anything unexpected, or would like to propose a new feature, please open an [issue](https://github.com/ComplexData-MILA/AIF-Gen/issues) and feel free [to discuss them with us](https://github.com/ComplexData-MILA/AIF-Gen/discussions).

To learn more about making a contribution to AIF-Gen see our [contribution guide](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
