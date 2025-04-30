#!/bin/bash
#SBATCH --job-name=validation_run
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

echo "Starting vLLM server..."
uvx vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype float16 --api-key $OPENAI_API_KEY &

# Save server process ID
SERVER_PID=$!

echo "Waiting for server to start..."
while true; do
  echo "Checking if server is up..."
  RESPONSE=$(curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" 2>&1)

  if [[ "$RESPONSE" == *"data"* ]]; then
    echo "Server is up and running!"
    break
  fi

  # Check if server is still running
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server process died unexpectedly"
    exit 1
  fi

  echo "Server not ready yet. Waiting 5 seconds..."
  sleep 5
done

# Run validation twice
echo "Running validation on llama70b generated data..."
uv run aif validate education_qna_hinted.json education_qna_hinted_70b_validation.json --no-validate-diversity --model meta-llama/Meta-Llama-3-8B-Instruct &

echo "Running validation on gpt-4o-mini generated data..."
uv run aif validate "$SCRATCH/education_qna_hinted.json" education_qna_hinted_4omini_validation.json --no-validate-diversity --model meta-llama/Meta-Llama-3-8B-Instruct &
wait

echo "Validation complete. Stopping the server"
kill $SERVER_PID

echo "All done!"
