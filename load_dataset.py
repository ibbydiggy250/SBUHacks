"""
Simple script to load ASL datasets into the database
"""

import os
import sys
from dataset_loader import ASLDatasetLoader

def main():
    print("=" * 60)
    print("ASL Dataset Loader")
    print("=" * 60)
    print()
    
    print("This script will load ASL signs from pre-existing images and videos.")
    print()
    print("Supported formats:")
    print("1. Folder structure: Each folder = sign name, contains images/videos")
    print("2. WLASL format: dataset/videos/SIGN_NAME/video_files")
    print("3. MS-ASL format: dataset/videos/ with MSASL_train.json")
    print("4. Flat structure: All files in one folder, named SIGN_NAME_001.jpg")
    print("5. JSON file: Custom JSON mapping signs to file paths")
    print()
    
    # Get dataset path
    dataset_path = input("Enter path to dataset folder or JSON file: ").strip().strip('"')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path does not exist: {dataset_path}")
        return
    
    # Detect format
    print("\nDetecting dataset format...")
    format_type = "auto"
    
    if dataset_path.endswith('.json'):
        format_type = "json"
        print("Detected: JSON format")
    elif os.path.exists(os.path.join(dataset_path, 'videos')):
        if os.path.exists(os.path.join(dataset_path, 'MSASL_train.json')):
            format_type = "msasl"
            print("Detected: MS-ASL format")
        else:
            format_type = "wlasl"
            print("Detected: WLASL format")
    else:
        # Check if it's flat or folder structure
        files = os.listdir(dataset_path)
        has_folders = any(os.path.isdir(os.path.join(dataset_path, f)) for f in files)
        
        if has_folders:
            format_type = "sign_name"
            print("Detected: Folder structure (each folder = sign name)")
        else:
            format_type = "flat"
            print("Detected: Flat structure (all files in one folder)")
    
    # Ask for confirmation
    print(f"\nFormat: {format_type}")
    confirm = input("Is this correct? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\nAvailable formats:")
        print("1. sign_name - Each folder = sign name")
        print("2. flat - All files in one folder")
        print("3. wlasl - WLASL dataset format")
        print("4. msasl - MS-ASL dataset format")
        print("5. json - JSON file format")
        format_choice = input("Enter format number (1-5): ").strip()
        
        format_map = {
            "1": "sign_name",
            "2": "flat",
            "3": "wlasl",
            "4": "msasl",
            "5": "json"
        }
        format_type = format_map.get(format_choice, format_type)
    
    # Output path
    output_path = input("\nOutput database path (default: asl_database.json): ").strip()
    if not output_path:
        output_path = "asl_database.json"
    
    # Media base path (for JSON format)
    media_base = None
    if format_type == "json":
        media_base = input("Base path for media files (leave empty if paths are absolute): ").strip()
        if not media_base:
            media_base = None
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    print("This may take a while depending on dataset size...")
    print()
    
    loader = ASLDatasetLoader()
    
    try:
        if format_type == "json":
            dataset = loader.load_from_custom_json(dataset_path, media_base)
            loader.save_to_database(dataset, output_path)
        else:
            loader.process_dataset(dataset_path, format_type, output_path)
        
        print("\n" + "=" * 60)
        print("Dataset loading complete!")
        print("=" * 60)
        
        # Show statistics
        from asl_database import ASLDatabase
        db = ASLDatabase(db_path=output_path)
        stats = db.get_statistics()
        
        print(f"\nDatabase Statistics:")
        print(f"  Total signs: {stats['num_signs']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"\nSigns loaded:")
        for sign, count in sorted(stats['signs'].items()):
            print(f"  {sign}: {count} samples")
        
    except KeyboardInterrupt:
        print("\n\nLoading interrupted by user.")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

