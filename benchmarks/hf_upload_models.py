from huggingface_hub import HfApi, upload_folder

datasets = 'aifgen-long-piecewise aifgen-lipschitz aifgen-piecewise-preference-shift aifgen-domain-preference-shift aifgen-short-piecewise CPPO-REWARD'
dataset_indices = '0 1 2 3 4 5 6 7 8 9'
model = 'Qwen2-0.5B'
# datasets="aifgen-long-piecewise"
# dataset_indices="0"

for dataset_name in datasets.split():
    for dataset_index in dataset_indices.split():
        # Upload the model to the Hugging Face Hub
        try:
            repo_id = f'LifelongAlignment/{model}-Instruct_{dataset_name}_REWARD_{dataset_index}'
            api = HfApi()
            api.create_repo(repo_id, repo_type='model', exist_ok=True, private=False)

            path = f'/lustre/orion/bif151/scratch/ivan.anokhin/AIF-Gen/{dataset_name}/{model}-Reward-8gpus/{model}-Instruct_{dataset_name}_REWARD_{dataset_index}'
            print('path', path)

            upload_folder(
                repo_id=repo_id,
                # path_in_repo=f"{dataset_name}-{dataset_index}/reward-model",
                folder_path=path,
                commit_message='Upload AIFGen reward model',
                repo_type='model',
            )
        except:
            print(f'Failed to upload {dataset_name}-{dataset_index} reward model')
            continue
