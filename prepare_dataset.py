"""
Dataset preparation script to convert various formats to ImageFolder format.

Supports:
1. Images in single folder with labels in filename
2. Images in single folder with metadata CSV
3. Unorganized images that need to be split and organized
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import csv
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT 1: Convert from filename labels
# ============================================================================

def prepare_from_filename_labels(
    source_dir: str,
    output_dir: str = "phone_damage",
    split_ratio: Tuple[float, float] = (0.8, 0.2),
    filename_pattern: str = None
):
    """
    Convert images with labels in filename to ImageFolder format.
    
    Assumes filenames like:
    - image_nodamage_001.jpg  (contains "nodamage" or "no_damage")
    - image_damage_001.jpg     (contains "damage")
    
    Args:
        source_dir: Directory containing all images
        output_dir: Output directory for ImageFolder structure
        split_ratio: (train_ratio, val_ratio)
        filename_pattern: Custom pattern to extract class from filename
                         Can contain keywords like "damage", "nodamage", "no_damage", "crack", "dent", "scratch"
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        for label in ["no_damage", "damage"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [f for f in source_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    logger.info(f"Found {len(image_files)} images in {source_dir}")
    
    # Classify and split
    train_images = []
    val_images = []
    
    for img_file in image_files:
        # Extract label from filename
        filename_lower = img_file.stem.lower()
        
        # Determine if damage or no_damage
        if any(word in filename_lower for word in ["nodamage", "no_damage", "undamaged", "intact"]):
            label = "no_damage"
        elif any(word in filename_lower for word in ["damage", "crack", "dent", "scratch", "broken", "cracked"]):
            label = "damage"
        else:
            logger.warning(f"Could not classify {img_file.name}, skipping")
            continue
        
        # Randomly assign to train or val
        if random.random() < split_ratio[0]:
            train_images.append((img_file, label))
        else:
            val_images.append((img_file, label))
    
    # Copy files
    total = 0
    for split_name, image_list in [("train", train_images), ("val", val_images)]:
        for img_file, label in image_list:
            dest = output_dir / split_name / label / img_file.name
            shutil.copy2(img_file, dest)
            total += 1
    
    logger.info(f"✓ Organized {total} images into {output_dir}")
    logger.info(f"  - Train: {len(train_images)} images")
    logger.info(f"  - Val:   {len(val_images)} images")


# ============================================================================
# FORMAT 2: Convert from CSV with labels
# ============================================================================

def prepare_from_csv(
    image_dir: str,
    csv_file: str,
    output_dir: str = "phone_damage",
    split_ratio: Tuple[float, float] = (0.8, 0.2),
    image_col: str = "image",
    label_col: str = "label"
):
    """
    Convert images with labels in CSV to ImageFolder format.
    
    CSV format:
        image,label
        image1.jpg,no_damage
        image2.jpg,damage
        image3.jpg,crack (will be merged to "damage")
    
    Args:
        image_dir: Directory containing images
        csv_file: CSV file with image names and labels
        output_dir: Output directory for ImageFolder structure
        split_ratio: (train_ratio, val_ratio)
        image_col: CSV column name with image filenames
        label_col: CSV column name with labels
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        for label in ["no_damage", "damage"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    images_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row[image_col].strip()
            label = row[label_col].strip().lower()
            
            # Normalize labels
            if label in ["no_damage", "nodamage", "undamaged", "intact"]:
                label = "no_damage"
            else:
                # All damage types merge to "damage"
                label = "damage"
            
            images_data.append((img_name, label))
    
    logger.info(f"Read {len(images_data)} entries from {csv_file}")
    
    # Shuffle and split
    random.shuffle(images_data)
    split_idx = int(len(images_data) * split_ratio[0])
    train_data = images_data[:split_idx]
    val_data = images_data[split_idx:]
    
    # Copy files
    total = 0
    for split_name, data_list in [("train", train_data), ("val", val_data)]:
        for img_name, label in data_list:
            src = image_dir / img_name
            if not src.exists():
                logger.warning(f"Image not found: {src}")
                continue
            
            dest = output_dir / split_name / label / img_name
            shutil.copy2(src, dest)
            total += 1
    
    logger.info(f"✓ Organized {total} images into {output_dir}")
    logger.info(f"  - Train: {len(train_data)} images")
    logger.info(f"  - Val:   {len(val_data)} images")


# ============================================================================
# FORMAT 3: Convert from existing folder structure
# ============================================================================

def prepare_from_folder_structure(
    source_dir: str,
    output_dir: str = "phone_damage",
    split_ratio: Tuple[float, float] = (0.8, 0.2),
    damage_keywords: List[str] = None,
    nodamage_keywords: List[str] = None
):
    """
    Reorganize images from arbitrary folder structure.
    
    Accepts folders with any names, tries to detect damage status from folder names.
    
    Example structures it can handle:
    source/
    ├── folder1/
    │   └── image1.jpg
    ├── folder2/
    │   └── image2.jpg
    
    Args:
        source_dir: Root directory containing image folders
        output_dir: Output directory for ImageFolder structure
        split_ratio: (train_ratio, val_ratio)
        damage_keywords: Keywords to detect damage folders
        nodamage_keywords: Keywords to detect no_damage folders
    """
    if damage_keywords is None:
        damage_keywords = ["damage", "crack", "dent", "scratch", "broken", "cracked", "damaged"]
    if nodamage_keywords is None:
        nodamage_keywords = ["no_damage", "nodamage", "good", "intact", "undamaged", "okay", "ok"]
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        for label in ["no_damage", "damage"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Collect all images with their labels
    images_data = []
    
    for folder in source_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name_lower = folder.name.lower()
        
        # Determine label from folder name
        if any(kw in folder_name_lower for kw in nodamage_keywords):
            label = "no_damage"
        elif any(kw in folder_name_lower for kw in damage_keywords):
            label = "damage"
        else:
            logger.warning(f"Could not classify folder: {folder.name}")
            continue
        
        # Get all images in folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                images_data.append((img_file, label))
    
    logger.info(f"Found {len(images_data)} images in folder structure")
    
    # Shuffle and split
    random.shuffle(images_data)
    split_idx = int(len(images_data) * split_ratio[0])
    train_data = images_data[:split_idx]
    val_data = images_data[split_idx:]
    
    # Copy files
    total = 0
    for split_name, data_list in [("train", train_data), ("val", val_data)]:
        for img_file, label in data_list:
            # Create unique filename to avoid conflicts
            dest = output_dir / split_name / label / img_file.name
            
            # If file already exists, add counter
            counter = 1
            base_name = img_file.stem
            ext = img_file.suffix
            while dest.exists():
                dest = output_dir / split_name / label / f"{base_name}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(img_file, dest)
            total += 1
    
    logger.info(f"✓ Organized {total} images into {output_dir}")
    logger.info(f"  - Train: {len(train_data)} images")
    logger.info(f"  - Val:   {len(val_data)} images")


# ============================================================================
# FORMAT 4: Manual folder assignment
# ============================================================================

def prepare_from_manual_folders(
    damage_folder: str,
    no_damage_folder: str,
    output_dir: str = "phone_damage",
    split_ratio: Tuple[float, float] = (0.8, 0.2)
):
    """
    Convert when you already have separate damage and no_damage folders.
    
    Args:
        damage_folder: Path to folder with damaged images
        no_damage_folder: Path to folder with undamaged images
        output_dir: Output directory for ImageFolder structure
        split_ratio: (train_ratio, val_ratio)
    """
    damage_folder = Path(damage_folder)
    no_damage_folder = Path(no_damage_folder)
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        for label in ["no_damage", "damage"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Collect images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    damage_images = [f for f in damage_folder.rglob('*') 
                     if f.is_file() and f.suffix.lower() in image_extensions]
    nodamage_images = [f for f in no_damage_folder.rglob('*') 
                       if f.is_file() and f.suffix.lower() in image_extensions]
    
    logger.info(f"Found {len(damage_images)} damaged images")
    logger.info(f"Found {len(nodamage_images)} no_damage images")
    
    # Combine and shuffle
    all_images = [(f, "damage") for f in damage_images] + \
                 [(f, "no_damage") for f in nodamage_images]
    random.shuffle(all_images)
    
    # Split
    split_idx = int(len(all_images) * split_ratio[0])
    train_data = all_images[:split_idx]
    val_data = all_images[split_idx:]
    
    # Copy files
    total = 0
    for split_name, data_list in [("train", train_data), ("val", val_data)]:
        for img_file, label in data_list:
            dest = output_dir / split_name / label / img_file.name
            
            # Handle duplicates
            counter = 1
            base_name = img_file.stem
            ext = img_file.suffix
            while dest.exists():
                dest = output_dir / split_name / label / f"{base_name}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(img_file, dest)
            total += 1
    
    logger.info(f"✓ Organized {total} images into {output_dir}")
    logger.info(f"  - Train: {len(train_data)} images")
    logger.info(f"  - Val:   {len(val_data)} images")


# ============================================================================
# Utility: Print dataset statistics
# ============================================================================

def print_dataset_stats(dataset_dir: str = "phone_damage"):
    """Print statistics about the organized dataset."""
    dataset_dir = Path(dataset_dir)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        split_dir = dataset_dir / split
        
        if not split_dir.exists():
            print(f"  Directory not found: {split_dir}")
            continue
        
        for label in ["no_damage", "damage"]:
            label_dir = split_dir / label
            count = len(list(label_dir.glob('*')))
            print(f"  {label}: {count} images")


# ============================================================================
# Main - Choose your conversion method
# ============================================================================

if __name__ == "__main__":
    """
    Choose ONE of the following based on your dataset format:
    """
    
    # ====== OPTION 1: Filename contains labels ======
    # Use if your images are named like: image_damage_001.jpg, image_nodamage_001.jpg
    print("="*60)
    print("OPTION 1: Convert from filename labels")
    print("="*60)
    print("Use this if images are in a single folder with labels in filename")
    print("Example: image_damage_001.jpg, image_nodamage_001.jpg")
    print("\nCode:")
    print("prepare_from_filename_labels(")
    print('    source_dir="path/to/images",')
    print('    output_dir="phone_damage",')
    print('    split_ratio=(0.8, 0.2)')
    print(")")
    
    # ====== OPTION 2: CSV file with labels ======
    # Use if you have a CSV with image names and labels
    print("\n" + "="*60)
    print("OPTION 2: Convert from CSV")
    print("="*60)
    print("Use this if you have a CSV file with image names and labels")
    print("CSV format:")
    print("  image,label")
    print("  image1.jpg,no_damage")
    print("  image2.jpg,damage")
    print("\nCode:")
    print("prepare_from_csv(")
    print('    image_dir="path/to/images",')
    print('    csv_file="labels.csv",')
    print('    output_dir="phone_damage"')
    print(")")
    
    # ====== OPTION 3: Existing folder structure ======
    # Use if images are already organized in subfolders
    print("\n" + "="*60)
    print("OPTION 3: Convert from folder structure")
    print("="*60)
    print("Use this if images are in subfolders")
    print("Example structure:")
    print("  source/")
    print("    ├── damage_images/")
    print("    ├── good_images/")
    print("    └── cracked_phones/")
    print("\nCode:")
    print("prepare_from_folder_structure(")
    print('    source_dir="path/to/source",')
    print('    output_dir="phone_damage"')
    print(")")
    
    # ====== OPTION 4: Already have separate folders ======
    # Use if you have separate damage and no_damage folders
    print("\n" + "="*60)
    print("OPTION 4: Convert from separate folders")
    print("="*60)
    print("Use this if you already have damage/ and no_damage/ folders")
    print("\nCode:")
    print("prepare_from_manual_folders(")
    print('    damage_folder="path/to/damaged",')
    print('    no_damage_folder="path/to/undamaged",')
    print('    output_dir="phone_damage"')
    print(")")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Uncomment the option that matches YOUR dataset format")
    print("2. Update the paths to point to your actual data")
    print("3. Run: python prepare_dataset.py")
    print("4. Check phone_damage/ folder structure is created correctly")
    print("="*60 + "\n")
    
    # Uncomment the option you need:
    
    # OPTION 1: Filename labels
    # prepare_from_filename_labels(
    #     source_dir="path/to/your/images",
    #     output_dir="phone_damage",
    #     split_ratio=(0.8, 0.2)
    # )
    
    # OPTION 2: CSV
    # prepare_from_csv(
    #     image_dir="path/to/your/images",
    #     csv_file="path/to/labels.csv",
    #     output_dir="phone_damage"
    # )
    
    # OPTION 3: Folder structure
    # prepare_from_folder_structure(
    #     source_dir="path/to/your/images",
    #     output_dir="phone_damage"
    # )
    
    # OPTION 4: Separate folders
    # prepare_from_manual_folders(
    #     damage_folder="path/to/damaged_images",
    #     no_damage_folder="path/to/undamaged_images",
    #     output_dir="phone_damage"
    # )
    
    # After conversion, print statistics
    # print_dataset_stats("phone_damage")
