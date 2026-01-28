"""
Batch inference and fraud detection script.
Processes multiple images or image pairs for fraud detection.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from phone_damage_detection import load_model, get_damage_score, detect_fraud


class PhoneDamageInference:
    """Helper class for batch inference and fraud detection."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.model, self.class_names = load_model(checkpoint_path, device=device)
        self.device = device
        print(f"Model loaded from {checkpoint_path}")
    
    def score_single_image(self, image_path: str) -> float:
        """
        Score a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            float: Damage probability (0–1)
        """
        return get_damage_score(image_path, self.model, device=self.device)
    
    def score_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Score multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of dicts with image path and damage score
        """
        results = []
        for i, img_path in enumerate(image_paths, 1):
            try:
                score = self.score_single_image(img_path)
                results.append({
                    'image': img_path,
                    'damage_score': round(score, 4),
                    'status': 'success'
                })
                print(f"[{i}/{len(image_paths)}] {Path(img_path).name}: {score:.4f}")
            except Exception as e:
                results.append({
                    'image': img_path,
                    'error': str(e),
                    'status': 'failed'
                })
                print(f"[{i}/{len(image_paths)}] {Path(img_path).name}: ERROR - {e}")
        
        return results
    
    def detect_fraud_pair(self, delivery_img: str, return_img: str, 
                         threshold: float = 0.15) -> Dict:
        """
        Detect fraud for a single delivery-return pair.
        
        Args:
            delivery_img: Path to delivery image
            return_img: Path to return image
            threshold: Fraud threshold
        
        Returns:
            Dict with fraud detection result
        """
        return detect_fraud(delivery_img, return_img, self.model, 
                          device=self.device, threshold=threshold)
    
    def detect_fraud_batch(self, image_pairs: List[Tuple[str, str]], 
                          threshold: float = 0.15) -> List[Dict]:
        """
        Detect fraud for multiple delivery-return pairs.
        
        Args:
            image_pairs: List of tuples (delivery_img, return_img)
            threshold: Fraud threshold
        
        Returns:
            List of fraud detection results
        """
        results = []
        for i, (delivery_img, return_img) in enumerate(image_pairs, 1):
            try:
                result = self.detect_fraud_pair(delivery_img, return_img, threshold)
                result['status'] = 'success'
                results.append(result)
                
                status = result['decision']
                print(f"[{i}/{len(image_pairs)}] {status} - "
                      f"D:{result['delivery_score']:.4f} R:{result['return_score']:.4f} "
                      f"Δ:{result['score_diff']:.4f}")
            except Exception as e:
                results.append({
                    'delivery_image': delivery_img,
                    'return_image': return_img,
                    'error': str(e),
                    'status': 'failed'
                })
                print(f"[{i}/{len(image_pairs)}] ERROR - {e}")
        
        return results


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def save_results_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    # Determine fieldnames from first result
    fieldnames = list(results[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


def load_image_pairs_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    Load delivery-return image pairs from CSV.
    
    CSV format:
        delivery_image,return_image
        path/to/delivery1.jpg,path/to/return1.jpg
        path/to/delivery2.jpg,path/to/return2.jpg
    """
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['delivery_image'], row['return_image']))
    return pairs


def load_image_paths_txt(txt_path: str) -> List[str]:
    """Load image paths from text file (one per line)."""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Initialize inference engine
    print("Initializing inference engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferencer = PhoneDamageInference(
        checkpoint_path="./checkpoints/best_model.pt",
        device=device
    )
    
    # Example 1: Score validation images
    print("\n" + "="*60)
    print("Example 1: Scoring Validation Images")
    print("="*60)
    
    val_damage = list(Path("phone_damage/val/damage").glob("*.jpg"))[:3]
    val_nodamage = list(Path("phone_damage/val/no_damage").glob("*.jpg"))[:3]
    
    if val_damage:
        print("\nDamaged images:")
        for img in val_damage:
            score = inferencer.score_single_image(str(img))
            print(f"  {img.name}: {score:.4f}")
    
    if val_nodamage:
        print("\nNo damage images:")
        for img in val_nodamage:
            score = inferencer.score_single_image(str(img))
            print(f"  {img.name}: {score:.4f}")
    
    # Example 2: Fraud detection
    print("\n" + "="*60)
    print("Example 2: Fraud Detection (Single Pair)")
    print("="*60)
    
    if val_damage and val_nodamage:
        delivery_img = str(val_nodamage[0])
        return_img = str(val_damage[0])
        
        print(f"\nDelivery (good): {Path(delivery_img).name}")
        print(f"Return (damaged): {Path(return_img).name}")
        print()
        
        fraud_result = inferencer.detect_fraud_pair(delivery_img, return_img)
        print(json.dumps(fraud_result, indent=2))
    
    # Example 3: Batch fraud detection from CSV (if you have one)
    print("\n" + "="*60)
    print("Example 3: Batch Fraud Detection from CSV")
    print("="*60)
    
    csv_file = "delivery_return_pairs.csv"
    if Path(csv_file).exists():
        print(f"\nProcessing {csv_file}...")
        pairs = load_image_pairs_csv(csv_file)
        fraud_results = inferencer.detect_fraud_batch(pairs, threshold=0.15)
        save_results_json(fraud_results, "fraud_results.json")
        save_results_csv(fraud_results, "fraud_results.csv")
        
        # Summary statistics
        successful = [r for r in fraud_results if r.get('status') == 'success']
        frauds = [r for r in successful if r.get('fraud_detected')]
        print(f"\nFraud Detection Summary:")
        print(f"Total cases: {len(fraud_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Fraud cases: {len(frauds)} ({100*len(frauds)/len(successful):.1f}%)")
    else:
        print(f"\n{csv_file} not found. Skipping batch processing.")
    
    print("\n" + "="*60)
    print("✓ Inference complete!")
    print("="*60)
