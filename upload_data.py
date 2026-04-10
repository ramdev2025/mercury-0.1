#!/usr/bin/env python3
"""
Upload dataset to Modal Volume for MercuryMoE.

Usage:
    python upload_data.py --data_dir data/ucf101
"""

import argparse
import modal


def main():
    parser = argparse.ArgumentParser(description="Upload data to Modal volume")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ucf101",
        help="Local path to dataset directory"
    )
    args = parser.parse_args()

    # Get or create the data volume
    print(f"📦 Connecting to Modal volume 'video-moe-data'...")
    data_volume = modal.Volume.from_name("video-moe-data", create_if_missing=True)

    # Upload the dataset
    print(f"⬆️  Uploading {args.data_dir} to volume...")
    try:
        data_volume.add_local_dir(args.data_dir, remote_path="/ucf101")
        data_volume.commit()
        print("✅ Data upload completed successfully!")
        
        # List uploaded files
        print("\n📁 Uploaded files:")
        for entry in data_volume.iterdir("/"):
            print(f"   /{entry.name}")
            
    except Exception as e:
        print(f"❌ Error uploading data: {e}")
        print("\nMake sure:")
        print("  1. You have run 'modal token new' to authenticate")
        print(f"  2. The directory '{args.data_dir}' exists locally")
        return

    print("\n💡 Next steps:")
    print("   Run training with: modal run modal_app.py")


if __name__ == "__main__":
    main()
