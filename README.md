<a id="readme-top"></a>

<div align="center">
  <h1 style="font-size:3vw;padding:0;margin:0;display:inline">AIF-Gen</h3>
  <h3 style="margin:0">A tool for generating synthetic, continual RLHF preference datasets</h3>
  <a href="https://aif_gen.readthedocs.io/en/latest"/><strong>Read the docs»</strong></a>
  <a href="https://github.com/ComplexData-MILA/AIF-Gen"/><strong>Read the paper»</strong></a>
</div>

<br/>

<div align="center">

<a href="">[![Contributors][contributors-shield]][contributors-url]</a>
<a href="">[![Issues][issues-shield]][issues-url]</a>
<a href="">[![MIT License][license-shield]][license-url]</a>

</div>

<div align="center">

<a href="">![example workflow](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/ruff.yml/badge.svg)</a>
<a href="">![example workflow](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/mypy.yml/badge.svg)</a>
<a href="">![example workflow](https://github.com/ComplexData-MILA/AIF-Gen/actions/workflows/testing.yml/badge.svg)</a>

</div>

## About The Project

_AIF-Gen_ is a Python library that generates (continual) RLHF preference datasets

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

To log the validation to Opik for automated evaluation, set the following environment variables:

```sh
export OPIK_BASE_URL="..."
export OPIK_PROJECT_NAME="..."

# optional for self-hosted installation
export OPIK_API_KEY="..."

# optionally, specify dataset name sent to Opik
export DATASET_NAME="education_qna_direct"
```

### Transform Data

```sh
uv run aif transform
```

## Installation

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

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

To learn more about making a contribution to _OpenDG_ see our [contribution page](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/ComplexData-MILA/AIF-Gen.svg?style=for-the-badge
[contributors-url]: https://github.com/ComplexData-MILA/AIF-Gen/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/ComplexData-MILA/AIF-Gen.svg?style=for-the-badge
[issues-url]: https://github.com/ComplexData-MILA/AIF-Gen/issues
[license-shield]: https://img.shields.io/github/license/ComplexData-MILA/AIF-Gen.svg?style=for-the-badge
[license-url]: https://github.com/ComplexData-MILA/AIF-Gen/blob/master/LICENSE.txt
