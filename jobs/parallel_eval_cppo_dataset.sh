# EVAL
datasets="CPPO-RL"
dataset_indices="0 1"
checkpoint_indices="300 1800 2100"

for dataset_index in $dataset_indices
do
  for dataset_name in $datasets
  do
      for checkpoint in $checkpoint_indices
      do
      job_name="${dataset_name}-${dataset_index}-${checkpoint}"
      mkdir -p out/
      run_cmd="jobs/schedule_eval_cppo_dataset.sh ${dataset_name} ${dataset_index} ${checkpoint}"
      sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
      cmd="$sbatch_cmd"
      echo -e "${cmd}"
      ${cmd}
      sleep 1
      done
  done
done