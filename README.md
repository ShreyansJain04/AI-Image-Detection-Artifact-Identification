# Deepfake Detection with LeViT and ConvNeXt

A robust framework for detecting deepfake images using state-of-the-art LeViT and ConvNeXt models. This repository includes scripts for training and testing the models on any compatible dataset, leveraging advanced techniques to enhance detection accuracy.

## Installation

### 1. Set Up the Conda Environment

Ensure you have Conda installed. Use the provided environment.yml to create the environment.

```bash
conda env create -f environment.yaml
conda activate deepfake-env
```

### 2. Dataset Preparation
Organize your dataset with the following structure:

```bash
dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ai/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── real/
│   └── ai/
└── test/
    ├── real/
    └── ai/
```

# Usage

## Training

Train the deepfake detection model using `train_levit.py`. Here is a example command to run the same:

```bash
python train_levit.py --data_dir /combined_dataset --model_name levit_192 --pretrained --save_dir ./combined --gpu_ids 0,1
```
To augment images, you can run the following command:
```bash
python augment_images.py "Input directory" "Output Dir"
```

## Testing

To test we will have to run the `test_levit.py` script. The example command follows:

```bash
python test_levit.py --image_dir /path/to/images \
                      --checkpoint_path /path/to/checkpoint.pth \
                      --output_json /path/to/output.json \
                      --batch_size 16 \
                      --gpu_ids 0
```

---
license: apache-2.0
---


# Artifact Explanation Using Ovis Model

## Overview

Taks2 leverages the **Ovis Model** to identify distinguishing artifacts in AI-generated images and provide interpretable explanations. By examining subtle irregularities and inconsistencies in image features, this model helps differentiate synthetic images from genuine ones while offering detailed reasoning for its classifications.

## Approach

We used the **Ovis Model**, a multimodal large language model (MLLM), to process both image and textual data. Our approach involves providing a single prompt describing a comprehensive list of artifacts to identify in AI-generated images. The model generates structured outputs explaining the detected artifacts and their positions in the image.

## Prompt Design

The single input prompt provided to the model included a detailed list of potential artifacts typically found in AI-generated images. Examples of artifacts include:

- Inconsistent object boundaries
- Texture bleeding between adjacent regions
- Unrealistic eye reflections
- Multiple light source conflicts
- Aliasing along high-contrast edges

The full prompt ensures exhaustive artifact detection and structured output in the format: 

```
{artifact name 1: explanation, artifact name 2: explanation, ...}
```

## Running the Script

### 1. Prepare Input Files
- `input.txt`: Contains the artifact detection prompt
- `64_task1.json`: JSON file with flagged images (results from Task 1)
- `perturbed_images_32/`: Directory containing images to analyze

### 2. Update Code Paths

#### Input File Path
```python
input_file = "input.txt" # Path to the artifact detection prompt
```

#### Output File Path
```python
output_file = "outputs/output.txt" # Path to store the output of artifact detection
```
note output.txt is a temp file.

#### JSON File Path
```python
json_file = "outputs/output.json" # Path to store structured artifact explanations
```

#### Task 1 JSON Path
```python
task1_json = "input/task1.json" # Path to the flagged images file from Task 1
```

#### Image Base Path
```python
image_base_path = "perturbed_images_32" # Path to the images for artifact detection
```

### 3. Model ID Configuration
```python
model_id = "AIDC-AI/Ovis1.6-Gemma2-9B" # Model ID from Hugging Face model
```

### 4. Execute the Script
```bash
python artifact_detection.py
```

## Methodology

### Step 1: Model Setup
We used the open-source **Ovis Model (v1.6 Gemma2-9B)**. The model was loaded with the following libraries:

- **Torch** (2.2.0): For efficient tensor computations.
- **Transformers** (4.44.2): To load and interact with the Ovis Model.
- **Numpy** (1.24.3): For numerical computations.
- **Pillow** (10.3.0): For image processing.

Install dependencies using:
```bash
pip install torch==2.2.0 transformers==4.44.2 numpy==1.24.3 pillow==10.3.0
```

### Step 2: Data Preprocessing
1. Images were stored in a designated directory (`perturbed_images_32`).
2. A JSON file (`task1.json`) identified images classified as "fake" and flagged them for processing.

### Step 3: Artifact Detection
For each flagged image:
1. The **image** and **textual prompt** were fed into the Ovis model.
2. The model analyzed the image and extrapolated explanations for the detected artifacts.
3. Output was parsed and stored as a structured JSON file (`task2.json`).

### Step 4: Explanation Generation
The model's output provided:
- Artifact names (e.g., *Inconsistent object boundaries*).
- Detailed explanations of why the artifacts make the image appear synthetic.
- Artifact positions in the image, enhancing interpretability.

## Ovis Model Configuration

### Loading Configuration
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    multimodal_max_length=8192,
    trust_remote_code=True
)
```

### Generation Configuration
```python
gen_kwargs = dict(
    max_new_tokens=2048,
    do_sample=False,
    top_p=None,
    top_k=None,
    temperature=None,
    repetition_penalty=None,
    eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=text_tokenizer.pad_token_id,
    use_cache=True
)
```

## Citation

```bibtex
@article{lu2024ovis,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model},
  author={Shiyin Lu and Yang Li and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang and Han-Jia Ye},
  year={2024},
  journal={arXiv:2405.20797}
}
```




