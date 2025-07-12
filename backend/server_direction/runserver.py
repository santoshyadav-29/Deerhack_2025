#!/usr/bin/env python3
"""
Simple script to run the Navigation Assistant Server
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from navigation_server import run_server
    
    if __name__ == "__main__":
        print("🚀 Starting Navigation Assistant Server...")
        print("📱 Open your browser to: http://localhost:8000")
        print("📋 Make sure your camera is connected and working")
        print("🔊 Ensure speakers/headphones are connected for audio")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        run_server(
            target_item='bottle',
            camera_id=1,
            host='0.0.0.0',  # Allow access from other devices on network
            port=8000
        )
        
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install required packages:")
    print("   pip install fastapi uvicorn opencv-python gtts")
except Exception as e:
    print(f"❌ Error starting server: {e}")
