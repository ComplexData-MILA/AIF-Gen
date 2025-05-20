source .env 
export HF_HUB_OFFLINE=0


# for name in dataset names create the model name based on : LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_{i}
# then add the i to the model name from 1 to 9 and use python and from huggingface_hub import snapshot_download to download the model

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='LifelongAlignment/CPPO-RL', revision='main', repo_type='dataset')"   

model_name="LifelongAlignment/Qwen2.5-0.5B-Instruct_CPPO_REWARD_0"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$model_name', revision='main')"
model_name="LifelongAlignment/Qwen2.5-0.5B-Instruct_CPPO_REWARD_1"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$model_name', revision='main')"

