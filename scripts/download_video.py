#!/usr/bin/env python3
"""
Download YouTube video to data/raw folder for training data ingestion.
Run this locally or use the Modal function below.
"""
import subprocess
import sys
import os

def download_video(youtube_url: str, output_path: str = "data/raw/music_video_01.mp4"):
    """Download a YouTube video using yt-dlp."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-check-certificates",
        youtube_url
    ]
    
    print(f"Downloading: {youtube_url}")
    print(f"Output: {output_path}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully downloaded to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Tip: If you get authentication errors, try:")
        print("   1. Download manually from YouTube")
        print("   2. Use --cookies-from-browser chrome (if you have browser cookies)")
        print("   3. Run this inside Modal environment instead")
        return False

if __name__ == "__main__":
    url = "https://youtu.be/mrV8kK5t0V8?si=aMMFQenekQJ8kL-8"
    output = "data/raw/music_video_01.mp4"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]
    
    success = download_video(url, output)
    sys.exit(0 if success else 1)
