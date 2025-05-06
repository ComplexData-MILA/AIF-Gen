# AIF-GEN Documentation

![image](./img/logo.svg)

AIF-Gen is a platform for generating synthetic RLHF datasets for lifelong reinforcement learning on LLMs.

Our main goal is to facilitate preference data generation at scale via [RL from AI feedback](https://arxiv.org/abs/2309.00267). AIF-Gen natively supports evolving preferences making it especially useful for studying non-stationary domains such as tutoring. Think of it like [Procgen](https://github.com/openai/procgen), but for RLHF.

### Library Highlights

- ⚡ Asynchronous LLM batch inference powered by [vLLM](https://github.com/vllm-project/vllm)
- 🔧 Modular prompt templates and fully customizable preference specification
- 🗄️ LLM response cache to avoid redundant API requests
- ✅ Validation metrics to judge synthetic data quality
- 🤗 Direct integration with HuggingFace for robust dataset management

## Installation:

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

For more details, see the [contribution guide](../.github/CONTRIBUTING.md).
