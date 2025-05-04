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

AIF-Gen is a Python library that generates (continual) RLHF preference datasets

![image](./docs/img/architecture-dark.svg#gh-dark-mode-only)
![image](./docs/img/architecture-light.svg#gh-light-mode-only)

### Library Highlights

## Quick Tour for New Users

We expose the following cli:

```sh
uv run aif
```

### Generating Data

- In this example, we run inference using [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf)
- The chat template we are using is found [here](https://github.com/ComplexData-MILA/AIF-Gen/blob/data/minimal_example/olmo-chat-template.jinja)
- We use the api-key `MY_KEY`, but anything works here
- This starts an inference server listening on `localhost:8000`

#### Install VLLM (only needs to be done once)

```sh
uv tool install vllm
```

#### Serve a model locally using VLLM

```sh
uvx --with setuptools serve allenai/OLMo-1B-hf --dtype auto --api-key MY_KEY --chat-template chat_templates/omlo-chat-template.jinja
```

#### Export env variables

```sh
export OPENAI_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=MY_KEY

# Optionally, set the following to cache OpenAI requests in Elasticsearch.
# export ELASTIC_SEARCH_HOST="..."
# export ELASTIC_SEARCH_API_KEY="..."
```

#### Generate some data (dry-run)

```sh
uv run aif generate config/aif_config.yaml allenai/OLMo-1B-hf --dry-run

# To ignore cache hit and update cache, set FORCE_CACHE_REFRESH=True .
# FORCE_CACHE_REFRESH=True uv run aif generate config/aif_config.yaml allenai/OLMo-1B-hf --dry-run
```

#### Generate some data (for real)

```sh
uv run aif generate config/aif_config.yaml allenai/OLMo-1B-hf
```

### Validating Data

```sh
uv run aif validate
```

### Transform Data

```sh
uv run aif transform
```

## Installation

The current recommended way to install AIF-Gen is from source.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```sh
# Create and activate your venv
uv venv my_venv --python 3.10 && source my_venv/bin/activate

# Install the wheels into the venv
uv pip install git+https://github.com/ComplexData-MILA/AIF-Gen.git

# Test the install
aif
```

### Without uv

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

To learn more about making a contribution to AIF-Gen see our [contribution page](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
