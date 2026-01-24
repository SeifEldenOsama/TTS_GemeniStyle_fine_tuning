import modal
import os
import subprocess
from pathlib import Path

GPU_CONFIG = "H100:1" 
NUM_GPUS = 1
VOLUME_NAME = "tts-dataset-storage"
MOUNT_PATH = Path("/data")
OUTPUT_DIR = MOUNT_PATH / "parler-tts-finetuned-h100-ultra-optimized"
HF_DATASET_REPO = "SeifElden2342532/parler-tts-dataset-format" 

REQUIREMENTS = [
    "torch==2.5.0",
    "torchaudio==2.5.0",
    "torchcodec==0.1",
    "accelerate",
    "datasets[audio]",
    "transformers==4.46.1",
    "pydantic==1.10.17",
    "tqdm",
    "soundfile",
    "scipy",
    "pyyaml",
    "protobuf==4.25.8",
    "wandb",
    "evaluate",
    "jiwer",
    "librosa",
    "bitsandbytes",
    "huggingface_hub",
    "parler-tts @ git+https://github.com/huggingface/parler-tts.git"
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11" )
    .apt_install("git", "ffmpeg", "libsndfile1") 
    .run_commands("ulimit -n 65536") 
    .pip_install(
        *REQUIREMENTS,
        extra_index_url="https://download.pytorch.org/whl/cu121"
     )
)

app = modal.App("parler-tts-h100-finetune-ultra-optimized", image=image)

@app.function(
    volumes={str(MOUNT_PATH): modal.Volume.from_name(VOLUME_NAME)},
    timeout=25000,
    gpu=GPU_CONFIG,
    env={"FORCE_LIBSNDFILE": "1"} 
)
def finetune_parler_tts():
    repo_path = Path("/root/parler-tts")
    if not repo_path.exists():
        print("Cloning Parler-TTS repository...")
        subprocess.run(["git", "clone", "https://github.com/huggingface/parler-tts.git", str(repo_path )], check=True)

    # Apply patches to Parler-TTS training code
    import training.data
    data_py_path = Path(training.data.__file__)
    
    with open(data_py_path, "r") as f:
        content = f.read()

    buggy_code = 'metadata_dataset_names = metadata_dataset_names.split("+") if metadata_dataset_names is not None else None'
    fixed_code = 'metadata_dataset_names = metadata_dataset_names.split("+") if (metadata_dataset_names is not None and isinstance(metadata_dataset_names, str)) else [None] * len(dataset_names)'
    if buggy_code in content:
        content = content.replace(buggy_code, fixed_code)

    buggy_eval_code = 'vectorized_datasets["validation"]'
    fixed_eval_code = 'vectorized_datasets["eval"]'
    if buggy_eval_code in content:
        content = content.replace(buggy_eval_code, fixed_eval_code)
    
    with open(data_py_path, "w") as f:
        f.write(content)

    training_script_path = repo_path / "training" / "run_parler_tts_training.py"
    with open(training_script_path, "r") as f:
        script_content = f.read()
    
    buggy_num_proc = 'num_proc=min(data_args.preprocessing_num_workers, len(vectorized_datasets["eval"]) - 1),'
    fixed_num_proc = 'num_proc=1,' 
    if buggy_num_proc in script_content:
        script_content = script_content.replace(buggy_num_proc, fixed_num_proc)

    with open(training_script_path, "w") as f:
        f.write(script_content)

    model_name = "parler-tts/parler-tts-mini-v1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_command = f"""
accelerate launch --num_processes={NUM_GPUS} training/run_parler_tts_training.py \\
    --model_name_or_path "{model_name}" \\
    --train_dataset_name "{HF_DATASET_REPO}" \\
    --train_dataset_config_name "default" \\
    --train_split_name "train" \\
    --eval_dataset_name "{HF_DATASET_REPO}" \\
    --eval_dataset_config_name "default" \\
    --eval_split_name "validation" \\
    --max_train_samples 1000 \\
    --max_eval_samples 100 \\
    --seed 42 \\
    --do_train true \\
    --do_eval true \\
    --preprocessing_num_workers 1 \\
    --evaluation_strategy "epoch" \\
    --description_column_name "description" \\
    --prompt_column_name "text" \\
    --target_audio_column_name "audio" \\
    --description_tokenizer_name "google/flan-t5-base" \\
    --prompt_tokenizer_name "google/flan-t5-base" \\
    --save_to_disk "/tmp/parler_dataset_processed" \\
    --temporary_save_to_disk "/tmp/parler_dataset_temp" \\
    --output_dir "{OUTPUT_DIR}" \\
    --overwrite_output_dir true \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 8 \\
    --gradient_accumulation_steps 2 \\
    --gradient_checkpointing true \\
    --optim "adamw_bnb_8bit" \\
    --max_steps 200 \\
    --bf16 true \\
    --report_to "none"
"""
    
    print(f"\n[STARTING] Starting ultra-optimized training on {NUM_GPUS} H100 GPU...")
    subprocess.run(training_command, shell=True, check=True, cwd=str(repo_path))
    modal.Volume.from_name(VOLUME_NAME).commit()
    print("\n[FINISHED] Fine-Tuning Complete!")

@app.local_entrypoint()
def main():
    finetune_parler_tts.remote()
