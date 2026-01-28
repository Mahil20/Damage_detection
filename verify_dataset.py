"""
Verify dataset structure - check if images are in correct folders.
"""

from pathlib import Path

def verify_dataset():
    """Check dataset structure."""
    phone_damage = Path("phone_damage")
    
    print("="*60)
    print("DATASET STRUCTURE VERIFICATION")
    print("="*60 + "\n")
    
    for split in ["train", "val"]:
        print(f"{split.upper()}:")
        split_dir = phone_damage / split
        
        for label in ["damage", "no_damage"]:
            label_dir = split_dir / label
            if label_dir.exists():
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
                images = []
                for ext in image_extensions:
                    images.extend(label_dir.glob(ext))
                
                print(f"  {label}: {len(images)} images")
                
                # Show first 3 image names
                if images:
                    print(f"    Examples: {', '.join([img.name for img in images[:3]])}")
            else:
                print(f"  {label}: NOT FOUND")
        print()
    
    print("="*60)
    print("Issue: If 'no_damage' folder has very few images or wrong images,")
    print("you may need to re-prepare the dataset.")
    print("="*60)

if __name__ == "__main__":
    verify_dataset()
