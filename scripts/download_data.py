import modal
import subprocess
import shutil
from pathlib import Path

app = modal.App("download-tts-fixed")
volume = modal.Volume.from_name("tts-dataset-storage", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install("kaggle"),
    secrets=[modal.Secret.from_name("kaggle-secret")],
    volumes={"/data": volume},
    timeout=3600 
)
def download_to_volume():
    """Download from Kaggle and structure for Parler-TTS"""
    target_dir = Path("/data/tts_dataset/teacher_dataset_large_updated/voices")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    temp_download_path = Path("/tmp/kaggle_data")
    temp_download_path.mkdir(exist_ok=True)
    
    print(f"Downloading Kaggle dataset to {temp_download_path}...")
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "seifosamahosney/tts-dataset",
        "-p", str(temp_download_path),
        "--unzip",
        "--force"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Kaggle Error: {result.stderr}")
        return f"Download failed: {result.stderr}"

    print("Download successful! Moving files to correct structure...")

    count = 0
    for file_path in temp_download_path.rglob("*.wav"):
        target_file = target_dir / file_path.name
        if not target_file.exists():
            shutil.move(str(file_path), str(target_file))
            count += 1
    
    print(f"Moved {count} .wav files to {target_dir}")
    volume.commit()
    return f"Data structured at {target_dir}"

@app.local_entrypoint()
def main():
    download_to_volume.remote()
