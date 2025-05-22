# EVAL
datasets="aifgen-long-piecewise"
dataset_indices="0 1"
checkpoint_indices="300 600"

for dataset_index in $dataset_indices
do
  for dataset_name in $datasets
  do
      for checkpoint in $checkpoint_indices
      do
      job_name="${dataset_name}-${dataset_index}-${checkpoint}"
      mkdir -p out/
      run_cmd="jobs/schedule_eval.sh ${dataset_name} ${dataset_index} ${checkpoint}"
      sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
      cmd="$sbatch_cmd"
      echo -e "${cmd}"
      ${cmd}
      sleep 1
      done
  done
done

datasets="aifgen-domain-preference-shift"
dataset_indices="0 1 2 3"
checkpoint_indices="300 531"

for dataset_index in $dataset_indices
do
  for dataset_name in $datasets
  do
      for checkpoint in $checkpoint_indices
      do
      job_name="${dataset_name}-${dataset_index}-${checkpoint}"
      mkdir -p out/
      run_cmd="jobs/schedule_eval.sh ${dataset_name} ${dataset_index} ${checkpoint}"
      sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
      cmd="$sbatch_cmd"
      echo -e "${cmd}"
      ${cmd}
      sleep 1
      done
  done
done

datasets="aifgen-lipschitz"
dataset_indices="0 1 2"
checkpoint_indices="300 900 1063"

for dataset_index in $dataset_indices
do
  for dataset_name in $datasets
  do
      for checkpoint in $checkpoint_indices
      do
      job_name="${dataset_name}-${dataset_index}-${checkpoint}"
      mkdir -p out/
      run_cmd="jobs/schedule_eval.sh ${dataset_name} ${dataset_index} ${checkpoint}"
      sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
      cmd="$sbatch_cmd"
      echo -e "${cmd}"
      ${cmd}
      sleep 1
      done
  done
done

datasets="aifgen-piecewise-preference-shift"
dataset_indices="0 1 2 3 4 5 6 7"
checkpoint_indices="300 1200 2100"

for dataset_index in $dataset_indices
do
  for dataset_name in $datasets
  do
      for checkpoint in $checkpoint_indices
      do
      job_name="${dataset_name}-${dataset_index}-${checkpoint}"
      mkdir -p out/
      run_cmd="jobs/schedule_eval.sh ${dataset_name} ${dataset_index} ${checkpoint}"
      sbatch_cmd="sbatch --job-name $job_name ${run_cmd}"
      cmd="$sbatch_cmd"
      echo -e "${cmd}"
      ${cmd}
      sleep 1
      done
  done
done