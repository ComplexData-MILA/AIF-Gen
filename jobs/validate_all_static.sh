#!/bin/bash
#SBATCH --job-name=validate_static_all
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load your env vars
source .env

# map each LLM to its endpoint and key

declare -A API_KEY=(
  ["gpt-4o-mini"]="$OPENAI_API_KEY"
)

LLMS=(
  "gpt-4o-mini"
)

FOLDERS=(
  "4omini_generation_downsampled"
  "70B_generation"
)

for llm in "${LLMS[@]}"; do
  unset OPENAI_BASE_URL
  export OPENAI_API_KEY="${API_KEY[$llm]}"

  for gen in "${FOLDERS[@]}"; do
    # derive validation folder name
    val_folder="${gen/_generation/_validation}"
    for sub in "data/$gen"/*; do
      [ -d "$sub" ] || continue

      infile="$sub/data.json"
      outdir="data/$val_folder/$llm/$(basename "$sub")"
      mkdir -p "$outdir"
      outfile="$outdir/validate.json"

      uv run aif validate \
        --max_concurrency 256 \
        "$infile" \
        "$outfile" \
        --no-validate-diversity \
        --no-validate-embedding-diversity \
        --model "$llm"
    done
  done
done
