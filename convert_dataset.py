"""
Convert your dataset from Damage/ and Normal/ folders to ImageFolder format.
"""

from prepare_dataset import prepare_from_manual_folders, print_dataset_stats

if __name__ == "__main__":
    print("Converting dataset from 'dataset main' format...")
    print()
    
    prepare_from_manual_folders(
        damage_folder="dataset main/Damage",
        no_damage_folder="dataset main/Normal",
        output_dir="phone_damage",
        split_ratio=(0.8, 0.2)
    )
    
    print()
    print_dataset_stats("phone_damage")
