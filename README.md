# 🗣️ Parler-TTS Fine-Tuning on Modal (H100 Optimized)

## Project Overview

This repository contains the infrastructure and scripts for fine-tuning the **Parler-TTS** model, a state-of-the-art Text-to-Speech (TTS) model from Hugging Face, to generate high-fidelity audio in a specific voice or style.

The entire workflow is designed for **ultra-optimized performance** and **reproducibility** using the [Modal](https://modal.com/) serverless cloud platform, specifically leveraging the immense power of the **NVIDIA H100 GPU** for accelerated training.

## Key Features

| Feature | Description | Benefit |
| :--- | :--- | :--- |
| **Parler-TTS Fine-Tuning** | Adapts the base Parler-TTS model to a custom voice dataset. | Generates high-quality, personalized speech. |
| **Modal Integration** | Scripts are written for seamless execution on the Modal platform. | Serverless, scalable, and reproducible cloud execution. |
| **NVIDIA H100 Optimization** | Training environment is configured to utilize the powerful H100 GPU. | Achieves significantly faster training times and handles larger models. |
| **Structured Codebase** | Notebook code is refactored into modular Python scripts. | Improves maintainability, professionalism, and ease of deployment. |

## Dataset

The fine-tuning process relies on a custom dataset. For this specific run, the model was **trained on 1,000 samples** and **evaluated on 100 samples**. The data preparation involves two main components: the raw audio data and the processed Hugging Face dataset repository.

| Component | Link | Description |
| :--- | :--- | :--- |
| **Raw Audio Data (Kaggle)** | [https://www.kaggle.com/datasets/seifosamahosney/tts-dataset](https://www.kaggle.com/datasets/seifosamahosney/tts-dataset) | The original Kaggle dataset containing the audio files used for fine-tuning. |
| **Hugging Face Dataset Repo** | `SeifElden2342532/parler-tts-dataset-format` | The pre-processed dataset, formatted and uploaded to the Hugging Face Hub for direct use in the training script. |

## Repository Structure

The original notebook logic has been modularized into a professional project structure:

```
.
├── notebooks/
│   ├── TTS_Gemeni_Style-fine-tuned.ipynb  # Original Modal notebook (for reference)
│   └── comparison_test/                   # Sample audio comparisons
├── scripts/
│   ├── download_data.py                   # Modal script to download and structure data from Kaggle
│   └── train.py                           # Modal script for H100-optimized Parler-TTS fine-tuning
├── src/
│   └── inference.py                       # Python script for running inference with the fine-tuned model
├── requirements.txt                       # Python dependencies for local development
└── README.md                              # This file
```

## Usage with Modal

### Prerequisites

1.  Install the Modal CLI: `pip install modal-client`
2.  Log in to Modal: `modal login`
3.  Create a Modal Secret named `kaggle-secret` with your Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`).

### 1. Data Preparation

The `download_data.py` script handles downloading the Kaggle dataset, unzipping it, and structuring it correctly into a persistent Modal Volume named `tts-dataset-storage`.

```bash
modal run scripts/download_data.py
```

### 2. Model Fine-Tuning (H100)

The `train.py` script executes the fine-tuning process. It is configured to request a dedicated **NVIDIA H100 GPU** instance on Modal for maximum training efficiency.

```bash
modal run scripts/train.py
```

The script automatically handles:
*   Setting up the CUDA environment.
*   Cloning the Parler-TTS repository.
*   Applying necessary patches for the training script.
*   Running the `accelerate launch` command with `bf16` and `adamw_bnb_8bit` for ultra-optimized training.
*   Saving the final model checkpoints to the persistent volume.

### 3. Inference

The `src/inference.py` script provides a template for loading the fine-tuned model and generating new audio.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Inference:**
    Update the `MODEL_PATH` in `src/inference.py` to point to your saved fine-tuned model, and then run:
    ```bash
    python src/inference.py
    ```

## Local Development

For local development and environment setup, the required dependencies are listed in `requirements.txt`. Note that the Modal-specific scripts (`download_data.py`, `train.py`) are designed to run exclusively on the Modal platform.

Developed with ❤️ by [Seif Elden Osama](https://github.com/SeifEldenOsama)
