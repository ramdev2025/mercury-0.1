# 📥 Download YouTube Video for Training

## Quick Start

### Option 1: Use Modal (Recommended)
Run the download directly in the Modal cloud environment:

```bash
modal run modal_app.py::download \
    --youtube-url "https://youtu.be/mrV8kK5t0V8?si=aMMFQenekQJ8kL-8" \
    --output-filename "music_video_01.mp4"
```

This will:
- ✅ Bypass local YouTube restrictions
- ✅ Save to your Modal data volume
- ✅ Be ready for immediate processing

### Option 2: Local Download
If you prefer to download locally first:

```bash
python scripts/download_video.py
```

Or with custom URL:
```bash
python scripts/download_video.py "https://youtu.be/YOUR_VIDEO_ID" "data/raw/my_video.mp4"
```

## Next Steps

After downloading, process the video for training:

```bash
# Ingest the video (extract transcripts, embeddings, etc.)
modal run modal_app.py::ingest_dataset --data-dir /workspace/data/raw
```

## Troubleshooting

### YouTube Authentication Errors
If you see "Sign in to confirm you're not a bot":
1. **Use Modal** (Option 1 above) - it often bypasses these restrictions
2. **Use browser cookies**:
   ```bash
   yt-dlp --cookies-from-browser chrome "YOUR_URL"
   ```
3. **Download manually** from YouTube and upload to Modal:
   ```bash
   modal volume put video-moe-data my_video.mp4 data/raw/
   ```

### Format Issues
To download in specific quality:
```bash
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" "URL"
```

## File Organization

```
data/raw/
├── music_video_01.mp4    # Your downloaded video
├── music_video_01.txt    # (Auto-generated transcript)
└── ...                   # Add more videos here
```

All videos in `data/raw/` will be processed together when you run `ingest_dataset`.
