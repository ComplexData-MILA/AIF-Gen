source .env 
export HF_HUB_OFFLINE=0

dataset_names=(
    "aifgen-domain-preference-shift"
    "aifgen-lipschitz"
    "aifgen-short-piecewise"
    "aifgen-long-piecewise"
    "aifgen-piecewise-preference-shift"
)

# for name in dataset names create the model name based on : LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_{i}
# then add the i to the model name from 1 to 9 and use python and from huggingface_hub import snapshot_download to download the model

for dataset_name in "${dataset_names[@]}"; do
    for i in {0..9}; do
        model_name="LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_${i}"
        data_name="LifelongAlignment/${dataset_name}"
        echo "Downloading model: $model_name"
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$data_name', revision='main', repo_type='dataset')"
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$model_name', revision='main')"
    done
done

