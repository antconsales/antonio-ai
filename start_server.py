#!/usr/bin/env python
"""
Start script for Antonio AI server with proper encoding support
"""
import sys
import os
from pathlib import Path

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set environment variable
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add FFmpeg to PATH
ffmpeg_path = Path(__file__).parent / "ffmpeg-8.0-essentials_build" / "bin"
if ffmpeg_path.exists():
    os.environ['PATH'] = str(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
    print(f"Added FFmpeg to PATH: {ffmpeg_path}")

# Now import and run uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
