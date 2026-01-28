"""
Direct folder-based fraud detection.
Upload delivery and return images to a folder and run detection.
"""

import torch
import json
from pathlib import Path
from phone_damage_detection import load_model, detect_fraud

# Configuration
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAUD_THRESHOLD = 0.15

def fraud_detection_from_folder(
    delivery_folder: str,
    return_folder: str,
    output_file: str = "fraud_results.json"
):
    """
    Run fraud detection on delivery and return images in separate folders.
    
    Folder structure:
    delivery/
    ├── order_001.jpg
    ├── order_002.jpg
    └── ...
    
    returns/
    ├── order_001.jpg
    ├── order_002.jpg
    └── ...
    
    Matches images by filename.
    """
    delivery_folder = Path(delivery_folder)
    return_folder = Path(return_folder)
    
    # Load model
    print("Loading trained model...")
    model, class_names = load_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    delivery_images = {
        f.stem: f for f in delivery_folder.glob('*')
        if f.suffix.lower() in image_extensions
    }
    
    return_images = {
        f.stem: f for f in return_folder.glob('*')
        if f.suffix.lower() in image_extensions
    }
    
    print(f"Found {len(delivery_images)} delivery images")
    print(f"Found {len(return_images)} return images\n")
    
    # Find matching pairs
    matching_ids = set(delivery_images.keys()) & set(return_images.keys())
    print(f"Found {len(matching_ids)} matching pairs\n")
    
    if not matching_ids:
        print("No matching image pairs found!")
        return
    
    # Run fraud detection
    results = []
    print("="*70)
    print("FRAUD DETECTION RESULTS")
    print("="*70)
    
    for i, img_id in enumerate(sorted(matching_ids), 1):
        delivery_img = str(delivery_images[img_id])
        return_img = str(return_images[img_id])
        
        try:
            result = detect_fraud(
                delivery_img, return_img, model,
                device=DEVICE,
                threshold=FRAUD_THRESHOLD
            )
            result['image_id'] = img_id
            results.append(result)
            
            # Print result
            status = "🚨 FRAUD" if result['fraud_detected'] else "✓ OK"
            print(f"\n[{i}/{len(matching_ids)}] {img_id}")
            print(f"  Delivery: {result['delivery_score']:.4f}")
            print(f"  Return:   {result['return_score']:.4f}")
            print(f"  Diff:     {result['score_diff']:.4f}")
            print(f"  Status:   {status}")
            
        except Exception as e:
            print(f"\n[{i}/{len(matching_ids)}] {img_id} - ERROR: {e}")
            results.append({
                'image_id': img_id,
                'error': str(e)
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if 'error' not in r]
    frauds = [r for r in successful if r.get('fraud_detected')]
    
    print(f"Total pairs:     {len(results)}")
    print(f"Processed:       {len(successful)}")
    print(f"Fraud detected:  {len(frauds)}")
    if successful:
        print(f"Fraud rate:      {100*len(frauds)/len(successful):.1f}%")
    print(f"Results saved:   {output_file}")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHONE DAMAGE FRAUD DETECTION - FOLDER MODE")
    print("="*70 + "\n")
    
    # Create example folders if they don't exist
    Path("delivery_images").mkdir(exist_ok=True)
    Path("return_images").mkdir(exist_ok=True)
    
    print("Instructions:")
    print("1. Place delivery images in: delivery_images/")
    print("2. Place return images in:   return_images/")
    print("3. Use same filename for matching pairs")
    print("   Example:")
    print("     delivery_images/order_001.jpg")
    print("     return_images/order_001.jpg")
    print()
    
    # Check if folders have images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    delivery_count = sum(len(list(Path("delivery_images").glob(ext))) for ext in image_extensions)
    return_count = sum(len(list(Path("return_images").glob(ext))) for ext in image_extensions)
    
    if delivery_count > 0 and return_count > 0:
        print(f"Found {delivery_count} delivery images and {return_count} return images")
        print("Running fraud detection...\n")
        
        fraud_detection_from_folder(
            delivery_folder="delivery_images",
            return_folder="return_images",
            output_file="fraud_results.json"
        )
    else:
        print(f"Waiting for images...")
        print(f"Delivery folder: {delivery_count} images")
        print(f"Return folder:   {return_count} images")
        print("\nAdd images to the folders and run this script again.")
