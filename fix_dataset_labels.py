"""
Fix reversed labels in dataset.
Swap damage and no_damage folders.
"""

import shutil
from pathlib import Path
import tempfile

def fix_reversed_labels():
    """Swap damage and no_damage labels."""
    phone_damage = Path("phone_damage")
    
    print("="*60)
    print("FIXING REVERSED LABELS")
    print("="*60 + "\n")
    
    for split in ["train", "val"]:
        split_dir = phone_damage / split
        
        damage_dir = split_dir / "damage"
        nodamage_dir = split_dir / "no_damage"
        
        if damage_dir.exists() and nodamage_dir.exists():
            print(f"\n{split.upper()} folder:")
            print(f"  Swapping 'damage' and 'no_damage'...")
            
            # Use temp folder for safe swap
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Move damage to temp
                temp_damage = temp_path / "temp_damage"
                shutil.move(str(damage_dir), str(temp_damage))
                print(f"    Moved 'damage' to temp")
                
                # Move no_damage to damage
                shutil.move(str(nodamage_dir), str(damage_dir))
                print(f"    Moved 'no_damage' → 'damage'")
                
                # Move temp to no_damage
                shutil.move(str(temp_damage), str(nodamage_dir))
                print(f"    Moved temp → 'no_damage'")
    
    print("\n" + "="*60)
    print("✓ Labels fixed! Damage and no_damage folders swapped.")
    print("="*60)
    
    # Verify
    print("\nVerification:")
    for split in ["train", "val"]:
        split_dir = phone_damage / split
        damage_count = len(list((split_dir / "damage").glob("*")))
        nodamage_count = len(list((split_dir / "no_damage").glob("*")))
        print(f"  {split}: {damage_count} damage, {nodamage_count} no_damage")

if __name__ == "__main__":
    fix_reversed_labels()
