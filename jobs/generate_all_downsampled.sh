#!/bin/bash
#SBATCH --job-name=generate_static_all_70B_final
#SBATCH --partition=main
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

# set -euo pipefail
source .env

# 1) start the vllm server in the background
uvx vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --dtype auto \
    --api-key openai \
    --tensor-parallel-size 2 &
SERVER_PID=$!
echo "⏳ Waiting for VLLM server (PID=$SERVER_PID) to come up…"

# replace fixed sleep with a health‐check loop
export UV_VLLM_SERVER_URL="http://127.0.0.1:8000"   # tell `uv run` where to send requests
for i in $(seq 1 600); do
  if curl -fs "${UV_VLLM_SERVER_URL}/health"; then
    echo "✅ VLLM up after $((i*5))s"
    break
  fi
  echo "…still waiting ($i/600)…"
  sleep 5
done

# helper to run one job
() { echo "➡️  $*"; eval "$*"; }


# 2) run all generation jobs sequentially
 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_qna_direct/data.json" \
    config/static_copy/education_qna_direct.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_qna_eli5/data.json" \
    config/static_copy/education_qna_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_qna_expert/data.json" \
    config/static_copy/education_qna_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_qna_hinted/data.json" \
    config/static_copy/education_qna_hinted.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_summary_eli5/data.json" \
    config/static_copy/education_summary_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/education_summary_expert/data.json" \
    config/static_copy/education_summary_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_generate_formal/data.json" \
    config/static_copy/politics_generate_formal.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_generate_rapper/data.json" \
    config/static_copy/politics_generate_rapper.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_generate_shakespeare/data.json" \
    config/static_copy/politics_generate_shakespeare.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_qna_eli5/data.json" \
    config/static_copy/politics_qna_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_qna_expert/data.json" \
    config/static_copy/politics_qna_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_summary_eli5/data.json" \
    config/static_copy/politics_summary_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/politics_summary_expert/data.json" \
    config/static_copy/politics_summary_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/tech_healthcare_qna_eli5/data.json" \
    config/static_copy/tech_healthcare_qna_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/tech_healthcare_qna_expert/data.json" \
    config/static_copy/tech_healthcare_qna_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/tech_physics_summary_eli5/data.json" \
    config/static_copy/tech_physics_summary_eli5.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/tech_physics_summary_expert/data.json" \
    config/static_copy/tech_physics_summary_expert.yaml \
    Meta-Llama-3.1-70B-Instruct

 uv run aif generate \
    --include-preference-axes \
    --max_concurrency 256 \
    --output_file "data/70B_generation/tech_physics_summary_highschool/data.json" \
    config/static_copy/tech_physics_summary_highschool.yaml \
    Meta-Llama-3.1-70B-Instruct

# 3) shutdown the server when done
echo "✅ All jobs finished. Shutting down VLLM server (PID=$SERVER_PID)…"
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo "🛑 Server stopped."
