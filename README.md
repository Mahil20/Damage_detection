# Phone Damage Detection Model

Image classification model for detecting phone damage and identifying return fraud.

## Features

- **Model**: EfficientNet-B3 pretrained on ImageNet
- **Input**: 224×224 RGB images
- **Output**: Damage probability score (0–1)
- **Fraud Detection**: Compare delivery vs. return images to flag suspicious returns
- **Training**: Automatic Mixed Precision (AMP) for faster training
- **Augmentation**: Resize, horizontal flip, brightness/contrast adjustments

## Dataset Preparation

### If your dataset is NOT in ImageFolder format, use `prepare_dataset.py`

This script converts common dataset formats to the required structure:

**Option 1: Filename contains labels**
```bash
# Images named: image_damage_001.jpg, image_nodamage_002.jpg
python prepare_dataset.py
# Then uncomment: prepare_from_filename_labels("path/to/images")
```

**Option 2: CSV file with labels**
```bash
# CSV format: image,label
# image1.jpg,no_damage
# image2.jpg,damage
python prepare_dataset.py
# Then uncomment: prepare_from_csv("path/to/images", "labels.csv")
```

**Option 3: Images in subfolders**
```bash
# Structure: source/damaged/, source/good/, source/cracked/
python prepare_dataset.py
# Then uncomment: prepare_from_folder_structure("path/to/images")
```

**Option 4: Already have separate damage/no_damage folders**
```bash
python prepare_dataset.py
# Then uncomment: prepare_from_manual_folders("path/to/damaged", "path/to/undamaged")
```

### Final Dataset Structure

After conversion, your dataset will be organized as:

```
phone_damage/
├── train/
│   ├── no_damage/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── damage/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── no_damage/
    │   ├── image1.jpg
    │   └── ...
    └── damage/
        ├── image1.jpg
        └── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

```python
from phone_damage_detection import train_model

model, class_names = train_model(
    dataset_root="phone_damage",
    output_dir="./checkpoints",
    epochs=15,
    batch_size=16,
    lr=1e-3
)
```

**Output**:
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/training_history.json` - Training metrics

### 2. Single Image Inference

```python
from phone_damage_detection import load_model, get_damage_score

model, class_names = load_model("checkpoints/best_model.pt")
damage_score = get_damage_score("path/to/image.jpg", model)

print(f"Damage probability: {damage_score:.4f}")
# Output: 0.8234 (82.34% likely damaged)
```

### 3. Fraud Detection

```python
from phone_damage_detection import load_model, detect_fraud

model, _ = load_model("checkpoints/best_model.pt")

result = detect_fraud(
    delivery_image_path="delivery.jpg",
    return_image_path="return.jpg",
    model=model,
    threshold=0.15
)

print(result)
# Output:
# {
#     'fraud_detected': True,
#     'delivery_score': 0.1200,
#     'return_score': 0.3500,
#     'score_diff': 0.2300,
#     'threshold': 0.15,
#     'decision': 'FRAUD'
# }
```

## Fraud Detection Logic

Given two images (delivery and return), the model computes damage scores:
- **Delivery Score (D)**: Damage probability at time of delivery
- **Return Score (R)**: Damage probability at time of return

**Decision Rule**:
```
If (R - D) > 0.15:
    FRAUD DETECTED (customer damaged phone after receiving it)
Else:
    OK RETURN (phone condition unchanged or improved)
```

## Model Architecture

```
EfficientNet-B3
├── Input: 224×224 RGB image
├── Backbone: ImageNet pretrained weights
├── Classifier: Linear(1536 → 2)
└── Output: [no_damage_prob, damage_prob]
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B3 |
| Input Size | 224×224 |
| Batch Size | 16 |
| Epochs | 15–20 |
| Optimizer | Adam (lr=1e-3) |
| Loss | CrossEntropyLoss |
| Mixed Precision | Enabled (AMP) |
| Device | CUDA (or CPU fallback) |

## Data Augmentation

| Transform | Parameters |
|-----------|-----------|
| Resize | 224×224 |
| Horizontal Flip | p=0.5 |
| Color Jitter | brightness=0.2, contrast=0.2 |
| Normalization | ImageNet stats |

## Output Files

After training, the following files are saved:

```
checkpoints/
├── best_model.pt          # Best model weights + metadata
└── training_history.json  # Training/validation metrics per epoch
```

## API Reference

### `train_model()`
Trains the model and saves checkpoints.

### `load_model(checkpoint_path, device)`
Loads a trained model from checkpoint.

### `get_damage_score(image_path, model, device)`
Returns damage probability (0–1) for a single image.

### `detect_fraud(delivery_img, return_img, model, threshold=0.15)`
Compares two images and returns fraud detection result.

### `create_data_loaders(dataset_root, batch_size, num_workers)`
Creates PyTorch DataLoaders with proper augmentation.

## Performance Notes

- **Training Time**: ~5–10 minutes per epoch on GPU (varies by dataset size)
- **Inference Time**: ~50–100ms per image (GPU)
- **Model Size**: ~12 MB
- **Memory**: ~4 GB VRAM for training with batch size 16

## Extending the Model

To use different thresholds or class configurations, modify:
- `threshold` parameter in `detect_fraud()` for different sensitivity
- `batch_size` in training for memory optimization
- `num_classes` in `create_model()` for multi-class damage types
