#!/bin/bash
#SBATCH --job-name=validate_static_all
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_qna_direct/*/data.json \
data/70B_15_validation/70B/education_qna_direct/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_qna_eli5/*/data.json \
data/70B_15_validation/70B/education_qna_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_qna_expert/*/data.json \
data/70B_15_validation/70B/education_qna_expert/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_qna_hinted/*/data.json \
data/70B_15_validation/70B/education_qna_hinted/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_summary_eli5/*/data.json \
data/70B_15_validation/70B/education_summary_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/education_summary_expert/*/data.json \
data/70B_15_validation/70B/education_summary_expert/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_generate_long/*/data.json \
data/70B_15_validation/70B/politics_generate_long/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_generate_short/*/data.json \
data/70B_15_validation/70B/politics_generate_short/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_qna_eli5/*/data.json \
data/70B_15_validation/70B/politics_qna_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_qna_expert/*/data.json \
data/70B_15_validation/70B/politics_qna_expert/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_summary_eli5/*/data.json \
data/70B_15_validation/70B/politics_summary_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_summary_expert/*/data.json \
data/70B_15_validation/70B/politics_summary_expert/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_generate_short/*/data.json \
data/70B_15_validation/70B/politics_generate_short/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_qna_eli5/*/data.json \
data/70B_15_validation/70B/politics_qna_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/politics_summary_eli5/*/data.json \
data/70B_15_validation/70B/politics_summary_eli5/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct \
&& \
uv run aif \
validate \
--max_concurrency 256 \
data/70B_15_generation/tech_healthcare_summary_expert/*/data.json \
data/70B_15_validation/70B/tech_healthcare_summary_expert/validate.json \
--no-validate-diversity \
--model Meta-Llama-3.1-70B-Instruct
