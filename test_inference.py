"""
Simple inference script to test the trained model.
"""

import torch
from pathlib import Path
from phone_damage_detection import load_model, get_damage_score, detect_fraud

# Configuration
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}\n")

# Load model
print("Loading trained model...")
model, class_names = load_model(CHECKPOINT_PATH, device=DEVICE)
print(f"Classes: {class_names}\n")

# ============================================================================
# TEST 1: Single Image Damage Score
# ============================================================================

print("="*60)
print("TEST 1: Get Damage Score for Single Image")
print("="*60)

# Test with a sample image from validation set
sample_images = [
    "phone_damage/val/damage/image1.jpg",
    "phone_damage/val/no_damage/image1.jpg",
]

for img_path in sample_images:
    img_file = Path(img_path)
    if img_file.exists():
        score = get_damage_score(img_path, model, device=DEVICE)
        status = "DAMAGED" if score > 0.5 else "NO DAMAGE"
        print(f"{img_path}")
        print(f"  Score: {score:.4f} → {status}\n")
    else:
        print(f"Image not found: {img_path}")

# ============================================================================
# TEST 2: Fraud Detection
# ============================================================================

print("="*60)
print("TEST 2: Fraud Detection (Delivery vs Return)")
print("="*60)

# Find actual test images
val_damage = list(Path("phone_damage/val/damage").glob("*.jpg"))
val_nodamage = list(Path("phone_damage/val/no_damage").glob("*.jpg"))

if val_damage and val_nodamage:
    delivery_img = str(val_nodamage[0])  # Good condition at delivery
    return_img = str(val_damage[0])      # Damaged at return
    
    print(f"Delivery image: {delivery_img}")
    print(f"Return image:   {return_img}\n")
    
    result = detect_fraud(delivery_img, return_img, model, device=DEVICE)
    
    print(f"Delivery Score:  {result['delivery_score']:.4f}")
    print(f"Return Score:    {result['return_score']:.4f}")
    print(f"Score Diff:      {result['score_diff']:.4f}")
    print(f"Threshold:       {result['threshold']:.4f}")
    print(f"\nDecision: {result['decision']}")
    
    if result['fraud_detected']:
        print("⚠️  FRAUD DETECTED - Customer likely damaged phone after delivery!")
    else:
        print("✓ OK RETURN - Phone condition unchanged or improved")
else:
    print("No test images found in validation set")

# ============================================================================
# TEST 3: Batch Inference
# ============================================================================

print("\n" + "="*60)
print("TEST 3: Batch Scoring")
print("="*60)

damage_imgs = list(Path("phone_damage/val/damage").glob("*.jpg"))[:3]
nodamage_imgs = list(Path("phone_damage/val/no_damage").glob("*.jpg"))[:3]

print(f"\nDamaged images ({len(damage_imgs)}):")
for img in damage_imgs:
    score = get_damage_score(str(img), model, device=DEVICE)
    print(f"  {img.name}: {score:.4f}")

print(f"\nNo damage images ({len(nodamage_imgs)}):")
for img in nodamage_imgs:
    score = get_damage_score(str(img), model, device=DEVICE)
    print(f"  {img.name}: {score:.4f}")

print("\n" + "="*60)
print("Model is ready for production!")
print("="*60)
print("\nNext steps:")
print("1. Use inference_batch.py for processing multiple image pairs")
print("2. Or integrate get_damage_score() and detect_fraud() into your app")
