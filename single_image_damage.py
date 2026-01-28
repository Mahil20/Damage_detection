"""
Simple single image damage detection.
Check if one image shows a damaged or undamaged phone.
"""

import torch
from pathlib import Path
from phone_damage_detection import load_model, get_damage_score

# Configuration
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_single_image(image_path: str):
    """
    Check if a single image shows damage or not.
    
    Args:
        image_path: Path to image file
    
    Returns:
        dict with damage info
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}
    
    # Load model
    model, class_names = load_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Get damage score
    damage_score = get_damage_score(str(image_path), model, device=DEVICE)
    
    # Determine if damaged or not
    # Threshold: 0.5 means 50% probability
    # Can adjust threshold if needed
    is_damaged = damage_score > 0.5
    
    result = {
        "image": image_path.name,
        "image_path": str(image_path),
        "damage_score": round(damage_score, 4),
        "is_damaged": is_damaged,
        "status": "DAMAGED ⚠️" if is_damaged else "NO DAMAGE ✓",
        "confidence": round(max(damage_score, 1 - damage_score) * 100, 2)
    }
    
    return result


if __name__ == "__main__":
    import sys
    
    # Example usage
    print("\n" + "="*60)
    print("SINGLE IMAGE DAMAGE DETECTION")
    print("="*60 + "\n")
    
    # Check if user provided image path as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = check_single_image(image_path)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Image:         {result['image']}")
            print(f"Damage Score:  {result['damage_score']}")
            print(f"Status:        {result['status']}")
            print(f"Confidence:    {result['confidence']}%")
    else:
        # Test with validation images
        print("Usage: python single_image_damage.py <image_path>")
        print("\nExample:")
        print("  python single_image_damage.py phone_damage/val/damage/image1.jpg")
        print("  python single_image_damage.py phone_damage/val/no_damage/image1.jpg\n")
        
        print("Testing with validation images:\n")
        
        import random
        
        # Test with damaged image
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        damaged_imgs = []
        for ext in image_extensions:
            damaged_imgs.extend(Path("phone_damage/val/damage").glob(ext))
        
        nodamage_imgs = []
        for ext in image_extensions:
            nodamage_imgs.extend(Path("phone_damage/val/no_damage").glob(ext))
        
        print(f"Found {len(damaged_imgs)} damaged images")
        print(f"Found {len(nodamage_imgs)} no_damage images\n")
        
        # Test multiple damaged images
        if damaged_imgs:
            test_damaged = random.sample(damaged_imgs, min(3, len(damaged_imgs)))
            print("Damaged Image Tests:")
            for img in test_damaged:
                result = check_single_image(str(img))
                if "error" not in result:
                    print(f"  {result['image']}: Score={result['damage_score']} → {result['status']}")
            print()
        
        # Test multiple no damage images
        if nodamage_imgs:
            test_nodamage = random.sample(nodamage_imgs, min(3, len(nodamage_imgs)))
            print("No Damage Image Tests:")
            for img in test_nodamage:
                result = check_single_image(str(img))
                if "error" not in result:
                    print(f"  {result['image']}: Score={result['damage_score']} → {result['status']}")
