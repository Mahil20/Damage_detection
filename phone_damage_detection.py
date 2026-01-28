import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def create_data_loaders(dataset_root, batch_size=16, num_workers=4):
    """
    Create DataLoaders for training and validation datasets.
    
    Args:
        dataset_root: Path to root directory containing train/ and val/ folders
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    dataset_root = Path(dataset_root)
    
    # Data augmentation pipeline
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    train_dataset = ImageFolder(
        root=dataset_root / "train",
        transform=train_transforms
    )
    
    val_dataset = ImageFolder(
        root=dataset_root / "val",
        transform=val_transforms
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(num_classes=2, device="cuda"):
    """
    Create EfficientNet-B3 model pretrained on ImageNet.
    
    Args:
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        model: EfficientNet-B3 model
    """
    # Load pretrained EfficientNet-B3
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # Replace final classification layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Automatic Mixed Precision
        with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(dataset_root, output_dir="./checkpoints", epochs=15, batch_size=16, lr=1e-3):
    """
    Full training pipeline.
    
    Args:
        dataset_root: Path to dataset
        output_dir: Directory to save checkpoints
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_root, batch_size=batch_size
    )
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes=len(class_names), device=device)
    
    # Loss, optimizer, scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "class_names": class_names
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining completed. Best Val Acc: {best_val_acc:.2f}%")
    return model, class_names


# ============================================================================
# INFERENCE
# ============================================================================

def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", ["no_damage", "damage"])
    
    model = create_model(num_classes=len(class_names), device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, class_names


def get_damage_score(image_path, model, device="cuda"):
    """
    Get damage probability score for a single image.
    
    Returns:
        float: Damage probability (0 to 1)
                0 = no damage, 1 = damaged
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and process image
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
    
    # Return damage probability (class 0 = "damage")
    # Classes are: [0]="damage", [1]="no_damage"
    damage_prob = probabilities[0, 0].item()
    return damage_prob


# ============================================================================
# FRAUD DETECTION
# ============================================================================

def detect_fraud(delivery_image_path, return_image_path, model, device="cuda", threshold=0.15):
    """
    Detect return fraud by comparing damage scores.
    
    Logic:
        If (return_damage - delivery_damage) > threshold:
            FRAUD DETECTED
        Else:
            OK RETURN
    
    Args:
        delivery_image_path: Path to delivery image
        return_image_path: Path to return image
        model: Trained model
        device: Device to run inference on
        threshold: Fraud threshold (default 0.15)
    
    Returns:
        dict: {
            'fraud_detected': bool,
            'delivery_score': float,
            'return_score': float,
            'score_diff': float,
            'threshold': float
        }
    """
    # Get damage scores
    delivery_score = get_damage_score(delivery_image_path, model, device)
    return_score = get_damage_score(return_image_path, model, device)
    
    # Calculate difference
    score_diff = return_score - delivery_score
    
    # Determine fraud
    fraud_detected = score_diff > threshold
    
    result = {
        'fraud_detected': fraud_detected,
        'delivery_score': round(delivery_score, 4),
        'return_score': round(return_score, 4),
        'score_diff': round(score_diff, 4),
        'threshold': threshold,
        'decision': 'FRAUD' if fraud_detected else 'OK RETURN'
    }
    
    return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = "phone_damage"  # Path to dataset with train/ and val/ folders
    CHECKPOINT_DIR = "./checkpoints"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    model, class_names = train_model(
        dataset_root=DATASET_ROOT,
        output_dir=CHECKPOINT_DIR,
        epochs=20,
        batch_size=16,
        lr=1e-3
    )
    
    print("\n✓ Training complete!")
    print(f"✓ Best model saved to: {CHECKPOINT_DIR}/best_model.pt")
    print(f"✓ Training history saved to: {CHECKPOINT_DIR}/training_history.json")
