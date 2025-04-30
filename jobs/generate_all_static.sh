#!/bin/bash
#SBATCH --job-name=generate_static_all_openai_final
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_qna_direct.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_qna_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_qna_expert.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_qna_hinted.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_summary_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/education_summary_expert.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_generate_rapper.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_generate_shakespeare.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_qna_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_qna_expert.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_summary_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/politics_summary_expert.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/tech_healthcare_qna_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/tech_healthcare_qna_expert.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/tech_physics_summary_eli5.yaml \
# gpt-4o-mini \
# && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/tech_physics_summary_expert.yaml \
# gpt-4o-mini && \
# uv run aif \
# generate \
# --max_concurrency 256 \
# config/static/tech_physics_summary_highschool.yaml \
# gpt-4o-mini

uv run aif \
generate \
--max_concurrency 256 \
config/static/politics_generate_rapper.yaml \
gpt-4o-mini \
&& \
uv run aif \
generate \
--max_concurrency 256 \
config/static/politics_generate_shakespeare.yaml \
gpt-4o-mini && \
uv run aif \
generate \
--max_concurrency 256 \
config/static/tech_physics_summary_highschool.yaml \
gpt-4o-mini
