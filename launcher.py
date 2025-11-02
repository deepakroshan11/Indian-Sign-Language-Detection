# launcher.py
# Unified launcher for ISL Emotion Translator System
# Starts both core processor and web UI in separate processes

import multiprocessing as mp
import time
import sys
import os

def start_core_processor(shared_state):
    """Start the core processing engine"""
    from isl_detection import core_processing_engine
    core_processing_engine(shared_state)

def start_web_ui(shared_state):
    """Start the web dashboard"""
    # Give core processor time to initialize
    time.sleep(2)
    from isl_ui_dashboard import start_ui_server
    start_ui_server(shared_state)

def main():
    """Main launcher"""
    print("=" * 70)
    print(" 🤟 ISL EMOTION TRANSLATOR - MULTI-PROCESS SYSTEM")
    print("=" * 70)
    print()
    print("🎯 Architecture:")
    print("   📹 Core Processor  → Handles CV, MediaPipe, ML (maximum speed)")
    print("   🌐 Web Dashboard   → Beautiful UI, real-time updates (no lag)")
    print("   🔗 IPC Queue       → Fast data transfer between processes")
    print()
    print("=" * 70)
    print()
    
    # Create shared state
    from isl_detection import SharedState
    shared_state = SharedState()
    
    # Start core processor process
    print("🚀 Starting Core Processor...")
    processor_proc = mp.Process(target=start_core_processor, args=(shared_state,))
    processor_proc.start()
    
    # Start web UI process
    print("🚀 Starting Web Dashboard...")
    ui_proc = mp.Process(target=start_web_ui, args=(shared_state,))
    ui_proc.start()
    
    print()
    print("=" * 70)
    print("✅ SYSTEM READY!")
    print("=" * 70)
    print()
    print("📊 Dashboard URL: http://localhost:5000")
    print()
    print("⌨️  Keyboard Shortcuts (in dashboard):")
    print("   • Ctrl+R        → Reset")
    print("   • Ctrl+Backspace → Backspace")
    print("   • 1/2/3         → Accept suggestions")
    print()
    print("🛑 Press Ctrl+C to stop all processes")
    print("=" * 70)
    
    try:
        # Keep main process alive
        while True:
            time.sleep(1)
            
            # Check if processes are still alive
            if not processor_proc.is_alive():
                print("\n⚠️  Core processor stopped!")
                break
            if not ui_proc.is_alive():
                print("\n⚠️  Web UI stopped!")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        # Send stop command
        try:
            shared_state.command_queue.put({'action': 'stop'})
        except:
            pass
        
        # Wait for graceful shutdown
        processor_proc.join(timeout=3)
        ui_proc.join(timeout=3)
        
        # Force terminate if needed
        if processor_proc.is_alive():
            processor_proc.terminate()
        if ui_proc.is_alive():
            ui_proc.terminate()
        
        print("✅ All processes stopped")
        print("=" * 70)

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
