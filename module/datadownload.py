import webdataset as wds
from huggingface_hub import HfApi, hf_hub_url
import json

hf_token = "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"
api = HfApi()
repo_id = "Amber-River/Pixiv-2.6M"
file_list = api.list_repo_files(repo_id, repo_type="dataset")
tar_files = [f for f in file_list if f.startswith("tags/data-") and f.endswith(".tar")]

urls = [
    f"pipe:curl -H \"Authorization: Bearer {hf_token}\" -L \"{hf_hub_url(repo_id, filename, repo_type='dataset')}\""
    for filename in tar_files
]

dataset = wds.WebDataset(urls, handler=wds.handlers.warn_and_continue, shardshuffle=False)

for i, sample in enumerate(dataset):
    print(f"Sample {i}:")
    for key, value in sample.items():
        if key == 'json':
            print(f"  {key}: {json.loads(value)}")
        else:
            print(f"  {key}: {value}")