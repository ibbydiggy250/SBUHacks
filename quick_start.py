"""
Quick Start Script - Load ASL datasets or collect training data
"""

import os
import sys

def main_menu():
    """Main menu for quick start"""
    print("=" * 60)
    print("ASL Translator - Quick Start")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Load from pre-existing ASL dataset (RECOMMENDED)")
    print("2. Collect training data from webcam")
    print("3. Exit")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        load_dataset_menu()
    elif choice == "2":
        collect_from_webcam()
    elif choice == "3":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")
        main_menu()

def load_dataset_menu():
    """Menu for loading datasets"""
    print("\n" + "=" * 60)
    print("Load ASL Dataset from Pre-existing Images/Videos")
    print("=" * 60)
    print("\nThis will load ASL signs from existing images and videos.")
    print("This is the RECOMMENDED method for better accuracy.")
    print()
    
    # Check if dataset_loader is available
    try:
        from load_dataset import main as load_dataset_main
        print("Starting dataset loader...")
        print()
        load_dataset_main()
    except ImportError:
        print("Error: dataset_loader module not found.")
        print("Please ensure all files are in the same directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def collect_from_webcam():
    """Collect training data from webcam"""
    print("\n" + "=" * 60)
    print("Collect Training Data from Webcam")
    print("=" * 60)
    print("\nNote: Loading from pre-existing datasets is recommended for better accuracy!")
    print("You'll need to show each sign to the camera multiple times.")
    print("\nRecommended: 20-30 samples per sign for better accuracy.\n")
    
    from asl_translator_app import ASLTranslatorApp
    
    # Initialize translator
    translator = ASLTranslatorApp(
        similarity_metric='combined',
        confidence_threshold=0.6
    )
    
    # Common ASL signs to collect
    common_signs = [
        "HELLO",
        "THANK YOU",
        "YES",
        "NO",
        "PLEASE",
        "SORRY",
        "HELP",
        "WATER",
        "FOOD",
        "GOOD",
        "BAD",
        "HAPPY",
        "SAD"
    ]
    
    # Letters A-Z
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    # Numbers 0-9
    numbers = [str(i) for i in range(10)]
    
    print("Choose what to collect:")
    print("1. Common phrases")
    print("2. Letters (A-Z)")
    print("3. Numbers (0-9)")
    print("4. Custom signs")
    print("5. All of the above")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    signs_to_collect = []
    
    if choice == "1":
        signs_to_collect = common_signs
    elif choice == "2":
        signs_to_collect = letters
    elif choice == "3":
        signs_to_collect = numbers
    elif choice == "4":
        custom = input("Enter signs (comma-separated): ").strip()
        signs_to_collect = [s.strip().upper() for s in custom.split(",")]
    elif choice == "5":
        signs_to_collect = common_signs + letters + numbers
    else:
        print("Invalid choice. Exiting.")
        return
    
    num_samples = int(input("Number of samples per sign (recommended: 20): ") or "20")
    
    print(f"\nYou will collect {num_samples} samples for {len(signs_to_collect)} signs.")
    input("Press Enter to start...")
    
    for i, sign in enumerate(signs_to_collect, 1):
        print(f"\n[{i}/{len(signs_to_collect)}] Collecting data for: {sign}")
        translator.collect_training_data(sign, num_samples)
        
        # Ask if user wants to continue
        if i < len(signs_to_collect):
            continue_choice = input("\nContinue to next sign? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
    
    print("\n" + "=" * 60)
    print("Training data collection complete!")
    print("=" * 60)
    
    # Show statistics
    stats = translator.database.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total signs: {stats['num_signs']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"\nSigns in database:")
    for sign, count in stats['signs'].items():
        print(f"  {sign}: {count} samples")
    
    print("\nYou can now run the translator:")
    print("  python asl_translator_app.py")
    print("  or")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
