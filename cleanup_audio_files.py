# cleanup_audio_files.py
# Run this to clean up leftover TTS audio files

import os
import glob

def cleanup_audio_files():
    """Remove all temporary voice files"""
    pattern = "voice_*.mp3"
    files = glob.glob(pattern)
    
    if not files:
        print("✅ No temporary audio files found")
        return
    
    print(f"🧹 Found {len(files)} temporary audio files")
    
    removed = 0
    failed = 0
    
    for file in files:
        try:
            os.remove(file)
            removed += 1
            print(f"  ✓ Removed: {file}")
        except Exception as e:
            failed += 1
            print(f"  ✗ Failed to remove {file}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Removed: {removed}")
    print(f"  ❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\n💡 Tip: Close the ISL application and try again")

if __name__ == "__main__":
    print("="*60)
    print("🧹 ISL Audio Cleanup Utility")
    print("="*60)
    cleanup_audio_files()
    print("="*60)