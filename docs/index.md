# AIF-GEN Documentation

![image](./img/logo.svg)

AIF-Gen is a platform for generating synthetic RLHF datasets for lifelong reinforcement learning on LLMs.

Our main goal is to facilitate preference data generation at scale via [RL from AI feedback](https://arxiv.org/abs/2309.00267). AIF-Gen natively supports evolving preferences making it especially useful for studying non-stationary domains such as tutoring. Think of it like [Procgen](https://github.com/openai/procgen), but for RLHF.

> \[!NOTE\]
> AIF-Gen is still alpha software, and may introduce breaking changes.

### Library Highlights

- ⚡ Asynchronous LLM batch inference powered by [vLLM](https://github.com/vllm-project/vllm)
- 🔧 Modular prompt templates and fully customizable preference specification
- 🗄️ LLM response cache to avoid redundant API requests
- ✅ Validation metrics to judge synthetic data quality
- 🤗 Direct integration with HuggingFace for robust dataset management

## Installing from source:

### Prerequisites

AIF-GEN uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment.
If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` installed globally you can just invoke:

```sh
pip install uv
```

### Clone the repository

```bash
git clone https://github.com/ComplexData-MILA/AIF-Gen.git
cd AIF-Gen
uv sync
```

For more details, see the [contribution guide](../.github/CONTRIBUTING.md).
