# isl_ui_dashboard.py
# Modern web-based UI for ISL translator
# Uses Flask + Socket.IO for real-time updates without blocking processing
# OPTIMIZED: Receives pre-processed data from core engine

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from multiprocessing import Queue
import cv2
import numpy as np
import json
import time
import base64
from threading import Thread, Lock

# Import shared state from core processor
import sys
sys.path.append('.')
from isl_detection import SharedState

app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-translator-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
shared_state = None
latest_data = {}
data_lock = Lock()

# -------------------------------
# ------ DATA RECEIVER ----------
# -------------------------------

def data_receiver_thread():
    """Receives data from core processor and broadcasts to web clients"""
    global latest_data, shared_state
    
    print("📡 Data receiver started...")
    
    while True:
        try:
            # Get data from processor (blocking with timeout)
            data = shared_state.ui_queue.get(timeout=1.0)
            
            with data_lock:
                # Decode frame from JPEG bytes
                frame_bytes = data['frame']
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert to base64 for web display
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Update latest data
                latest_data = {
                    'frame': frame_b64,
                    'current_word': data['current_word'],
                    'sentence': data['sentence'],
                    'suggestions': data['suggestions'],
                    'emotion': data['emotion'],
                    'emotion_scores': data['emotion_scores'],
                    'emotion_timeline': data['emotion_timeline'],
                    'stats': data['stats'],
                    'detected_letter': data['detected_letter'],
                    'hand_detected': data['hand_detected'],
                    'fps': round(data['fps'], 1),
                    'timestamp': data['timestamp']
                }
            
            # Broadcast to all connected clients
            socketio.emit('update', latest_data, namespace='/isl')
            
        except Exception as e:
            time.sleep(0.1)
            continue

# -------------------------------
# -------- ROUTES ---------------
# -------------------------------

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('dashboard.html')

@socketio.on('connect', namespace='/isl')
def handle_connect():
    """Client connected"""
    print('🔌 Client connected')
    # Send current state immediately
    with data_lock:
        if latest_data:
            emit('update', latest_data)

@socketio.on('command', namespace='/isl')
def handle_command(data):
    """Handle commands from UI"""
    global shared_state
    
    action = data.get('action')
    print(f'🎮 Command: {action}')
    
    try:
        if action == 'reset':
            shared_state.command_queue.put({'action': 'reset'})
            emit('feedback', {'status': 'success', 'message': 'Reset done'})
        
        elif action == 'backspace':
            shared_state.command_queue.put({'action': 'backspace'})
            emit('feedback', {'status': 'success', 'message': 'Backspace'})
        
        elif action == 'accept_suggestion':
            word = data.get('word', '')
            shared_state.command_queue.put({'action': 'accept_suggestion', 'word': word})
            emit('feedback', {'status': 'success', 'message': f'Accepted: {word}'})
        
        elif action == 'export':
            # Handle export (save to file)
            emit('feedback', {'status': 'success', 'message': 'Export not yet implemented'})
        
        else:
            emit('feedback', {'status': 'error', 'message': 'Unknown command'})
    except Exception as e:
        emit('feedback', {'status': 'error', 'message': str(e)})

# -------------------------------
# ---------- MAIN ---------------
# -------------------------------

def start_ui_server(shared_state_obj):
    """Start the web UI server"""
    global shared_state
    shared_state = shared_state_obj
    
    # Start data receiver thread
    receiver = Thread(target=data_receiver_thread, daemon=True)
    receiver.start()
    
    print("\n" + "="*60)
    print("🌐 ISL Web Dashboard Starting...")
    print("="*60)
    print("📊 Dashboard URL: http://localhost:5000")
    print("🔥 Real-time updates via WebSocket")
    print("="*60 + "\n")
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    # This should be run after starting the core processor
    print("⚠️  Make sure isl_core_processor.py is running first!")
    print("Loading shared state...")
    
    shared_state = SharedState()
    start_ui_server(shared_state)