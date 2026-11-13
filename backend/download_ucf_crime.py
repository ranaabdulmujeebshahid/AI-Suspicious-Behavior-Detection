import os
from huggingface_hub import snapshot_download

target_dir = "/content/drive/MyDrive/AI_Final_Project/backend/dataset/UCF_Crime"
os.makedirs(target_dir, exist_ok=True)

print("Starting UCF-Crime download (~103 GB) into Google Drive...")
snapshot_download(
    repo_id="jinmang2/ucf_crime",
    repo_type="dataset",
    local_dir=target_dir
)
print("UCF-Crime download complete!")
