"""
FastAPI Navigation Server
Real-time object navigation with natural language voice instructions
"""

import cv2
import asyncio
import json
import time
import threading
import os
import uuid
from queue import Queue, Empty
from typing import Optional, Dict, Any, List
import base64
import numpy as np
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our navigation system
from main import RobustObjectNavigationSystem

# Text-to-speech for natural voice output (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

# Google Text-to-Speech for web-based audio
try:
    from gtts import gTTS
    import uuid
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. Install with: pip install gtts")

# Google Translate for multilingual support
try:
    from googletrans import Translator
    from io import BytesIO
    TRANSLATE_AVAILABLE = True
except ImportError:
    TRANSLATE_AVAILABLE = False
    print("Warning: googletrans not available. Install with: pip install googletrans==4.0.0-rc1")

# Data models
class NavigationInstruction(BaseModel):
    instruction: str
    natural_language: str
    confidence: float
    timestamp: str
    person_detected: bool
    target_detected: bool
    distance_to_target: Optional[float]
    facing_target: bool
    target_direction: str

class SystemStatus(BaseModel):
    active: bool
    camera_connected: bool
    target_item: str
    person_locked: bool
    target_locked: bool
    last_update: str

# TTS request model for gTTS endpoints
class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"

class ObjectTrackingBuffer:
    """
    Buffer system to track object positions over time and guide users to last known locations
    """
    def __init__(self, buffer_file: str = "object_tracking_buffer.json", max_age_minutes: int = 30):
        self.buffer_file = buffer_file
        self.max_age_seconds = max_age_minutes * 60
        self.tracked_objects = {}
        self.load_buffer()
    
    def load_buffer(self):
        """Load existing buffer from JSON file"""
        try:
            if os.path.exists(self.buffer_file):
                with open(self.buffer_file, 'r') as f:
                    data = json.load(f)
                    self.tracked_objects = data.get('tracked_objects', {})
                    print(f"✅ Loaded {len(self.tracked_objects)} tracked objects from buffer")
            else:
                self.tracked_objects = {}
                print("📝 Created new object tracking buffer")
        except Exception as e:
            print(f"⚠️ Error loading buffer: {e}")
            self.tracked_objects = {}
    
    def save_buffer(self):
        """Save buffer to JSON file"""
        try:
            buffer_data = {
                'tracked_objects': self.tracked_objects,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.buffer_file, 'w') as f:
                json.dump(buffer_data, f, indent=2)
            
            print(f"💾 Saved buffer with {len(self.tracked_objects)} objects")
        except Exception as e:
            print(f"❌ Error saving buffer: {e}")
    
    def update_object_position(self, object_class: str, bbox: tuple, confidence: float, frame_shape: tuple, depth_estimate: float = None):
        """
        Update the position of a detected object
        
        Args:
            object_class: Class name of the object (e.g., 'bottle', 'person')
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence
            frame_shape: Shape of the frame (height, width)
            depth_estimate: Estimated depth in centimeters
        """
        current_time = time.time()
        
        # Calculate normalized position (0-1 range for frame independence)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / frame_shape[1]  # Normalize by width
        center_y = (y1 + y2) / 2 / frame_shape[0]  # Normalize by height
        
        # Calculate relative size
        width_ratio = (x2 - x1) / frame_shape[1]
        height_ratio = (y2 - y1) / frame_shape[0]
        
        object_data = {
            'class': object_class,
            'last_seen': current_time,
            'timestamp': datetime.now().isoformat(),
            'position': {
                'center_x': center_x,
                'center_y': center_y,
                'width_ratio': width_ratio,
                'height_ratio': height_ratio
            },
            'confidence': confidence,
            'depth_estimate': depth_estimate,
            'frame_shape': frame_shape,
            'bbox_absolute': list(bbox),  # Store absolute coordinates for reference
            'detection_count': self.tracked_objects.get(object_class, {}).get('detection_count', 0) + 1
        }
        
        self.tracked_objects[object_class] = object_data
        print(f"📍 Updated position for {object_class} (confidence: {confidence:.2f})")
    
    def get_last_known_position(self, object_class: str) -> Optional[Dict]:
        """
        Get the last known position of an object
        
        Args:
            object_class: Class name of the object
            
        Returns:
            Dictionary with position data or None if not found/expired
        """
        if object_class not in self.tracked_objects:
            return None
        
        object_data = self.tracked_objects[object_class]
        current_time = time.time()
        
        # Check if the data is still valid (not too old)
        if current_time - object_data['last_seen'] > self.max_age_seconds:
            print(f"⏰ Position data for {object_class} is too old, removing from buffer")
            del self.tracked_objects[object_class]
            return None
        
        return object_data
    
    def cleanup_old_entries(self):
        """Remove old entries from the buffer"""
        current_time = time.time()
        to_remove = []
        
        for object_class, object_data in self.tracked_objects.items():
            if current_time - object_data['last_seen'] > self.max_age_seconds:
                to_remove.append(object_class)
        
        for object_class in to_remove:
            del self.tracked_objects[object_class]
            print(f"🗑️ Removed expired entry for {object_class}")
        
        if to_remove:
            self.save_buffer()
    
    def get_all_tracked_objects(self) -> Dict:
        """Get all currently tracked objects"""
        self.cleanup_old_entries()
        return self.tracked_objects.copy()
    
    def calculate_navigation_to_last_position(self, target_class: str, current_frame_shape: tuple) -> Optional[Dict]:
        """
        Calculate navigation instructions to guide user to last known position of target
        
        Args:
            target_class: Class name of the target object
            current_frame_shape: Current frame dimensions
            
        Returns:
            Navigation instruction dict or None if no valid position
        """
        last_position = self.get_last_known_position(target_class)
        if not last_position:
            return None
        
        pos = last_position['position']
        
        # Convert normalized coordinates back to current frame
        target_center_x = pos['center_x'] * current_frame_shape[1]
        target_center_y = pos['center_y'] * current_frame_shape[0]
        
        # Calculate frame center
        frame_center_x = current_frame_shape[1] / 2
        frame_center_y = current_frame_shape[0] / 2
        
        # Calculate relative position
        dx = target_center_x - frame_center_x
        dy = target_center_y - frame_center_y
        
        # Generate navigation instruction
        horizontal_instruction = ""
        vertical_instruction = ""
        
        # Horizontal guidance
        if abs(dx) > current_frame_shape[1] * 0.1:  # 10% threshold
            if dx > 0:
                horizontal_instruction = "turn right"
            else:
                horizontal_instruction = "turn left"
        
        # Vertical guidance
        if abs(dy) > current_frame_shape[0] * 0.1:  # 10% threshold
            if dy > 0:
                vertical_instruction = "look down"
            else:
                vertical_instruction = "look up"
        
        # Combine instructions
        instructions = []
        if horizontal_instruction:
            instructions.append(horizontal_instruction)
        if vertical_instruction:
            instructions.append(vertical_instruction)
        
        if not instructions:
            instruction = "The target should be right in front of you"
        else:
            instruction = "To find the target, " + " and ".join(instructions)
        
        # Calculate approximate age of the data
        age_minutes = (time.time() - last_position['last_seen']) / 60
        
        return {
            'instruction': instruction,
            'confidence': max(0.3, 0.8 - (age_minutes * 0.1)),  # Decrease confidence with age
            'last_seen_minutes_ago': age_minutes,
            'position_data': last_position,
            'target_direction': horizontal_instruction if horizontal_instruction else "ahead"
        }

class NavigationServer:
    def __init__(self, target_item: str = 'bottle', camera_id: int = 0):
        """Initialize the navigation server"""
        self.app = FastAPI(title="Navigation Assistant", version="1.0.0")
        
        # Enable CORS for frontend access - Allow all requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_credentials=False,  # Set to False when using wildcard origins
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],  # All HTTP methods
            allow_headers=["*"],  # Allow all headers
            expose_headers=["*"],  # Expose all headers to frontend
        )
        
        self.navigation_system = RobustObjectNavigationSystem(target_item=target_item)
        self.camera_id = camera_id
        
        # Create audio directory for gTTS files
        self.audio_dir = "audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Track latest auto-generated audio file
        self.latest_auto_audio = None
        
        # Mount static files directory for serving HTML and assets
        static_dir = os.path.dirname(__file__)
        if os.path.exists(static_dir):
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Threading and communication
        self.frame_queue = Queue(maxsize=2)
        self.instruction_queue = Queue(maxsize=10)
        
        # Stabilization for high FPS sensitivity
        self.instruction_history = []
        self.max_history_size = 5  # Store last 5 instructions for stabilization
        self.last_stable_instruction = None
        self.current_instruction = NavigationInstruction(
            instruction="Initializing...",
            natural_language="Getting ready to help you navigate.",
            confidence=0.0,
            timestamp=datetime.now().isoformat(),
            person_detected=False,
            target_detected=False,
            distance_to_target=None,
            facing_target=False,
            target_direction="unknown"
        )
        
        # Camera and processing state
        self.cap = None
        self.processing_active = False
        self.camera_thread = None
        self.processing_thread = None
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # TTS setup - Use Google TTS for everything
        self.tts_enabled = GTTS_AVAILABLE
        if self.tts_enabled:
            print("Google TTS available for web audio generation")
        else:
            print("Warning: Google TTS not available. TTS features disabled.")
        
        # Translation setup
        self.translate_enabled = TRANSLATE_AVAILABLE
        if self.translate_enabled:
            self.translator = Translator()
        
        # Object Tracking Buffer System
        self.object_buffer = ObjectTrackingBuffer()
        self.buffer_update_interval = 3.0  # Update buffer every 3 seconds
        self.last_buffer_update = 0
        self.object_tracking_active = True
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def get_index():
            """Serve the main web interface"""
            try:
                # Try to serve the external HTML file
                html_path = os.path.join(os.path.dirname(__file__), "index.html")
                if os.path.exists(html_path):
                    return FileResponse(html_path, media_type="text/html")
                else:
                    # Fallback to basic interface
                    return HTMLResponse("<h1>Navigation Assistant</h1><p>Please create index.html file</p>")
            except Exception as e:
                return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>")
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status"""
            return SystemStatus(
                active=self.processing_active,
                camera_connected=self.cap is not None and self.cap.isOpened(),
                target_item=self.navigation_system.target_item,
                person_locked=self.navigation_system.person_tracker.locked_person_id is not None,
                target_locked=self.navigation_system.target_tracker.locked_target_id is not None,
                last_update=datetime.now().isoformat()
            )
        
        @self.app.get("/api/instruction")
        async def get_current_instruction():
            """Get the current navigation instruction"""
            return self.current_instruction
        
        @self.app.post("/api/start")
        async def start_navigation():
            """Start the navigation system"""
            try:
                await self.start_camera_processing()
                return {"status": "started", "message": "Navigation system started successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/stop")
        async def stop_navigation():
            """Stop the navigation system"""
            try:
                await self.stop_camera_processing()
                return {"status": "stopped", "message": "Navigation system stopped"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/targets")
        async def get_available_targets():
            """Get list of available target items"""
            try:
                available_items = list(self.navigation_system.object_dimensions.keys())
                return {
                    "available_targets": available_items,
                    "current_target": self.navigation_system.target_item,
                    "total_count": len(available_items),
                    "categories": {
                        "electronics": ["laptop", "cell phone", "tv", "mouse", "keyboard", "remote"],
                        "furniture": ["chair", "couch", "bed", "dining table"],
                        "kitchen": ["bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "microwave", "oven", "toaster", "sink", "refrigerator"],
                        "personal": ["backpack", "handbag", "suitcase", "book", "clock", "scissors"],
                        "other": ["potted plant", "toilet", "vase", "teddy bear", "hair drier", "toothbrush"]
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/change_target/{target_item}")
        async def change_target(target_item: str):
            """Change the target item to track"""
            try:
                available_items = list(self.navigation_system.object_dimensions.keys())
                if target_item.lower() not in available_items:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid target. Available: {available_items}"
                    )
                
                self.navigation_system.target_item = target_item.lower()
                self.navigation_system.reset_tracking()
                
                return {
                    "status": "success", 
                    "message": f"Target changed to {target_item}",
                    "target": target_item,
                    "previous_target": getattr(self, '_previous_target', 'unknown')
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/reset")
        async def reset_tracking():
            """Reset all tracking data"""
            try:
                self.navigation_system.reset_tracking()
                return {"status": "success", "message": "Tracking data reset"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/video_feed")
        async def video_feed():
            """Stream processed video feed"""
            return StreamingResponse(
                self.generate_video_stream(), 
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        # Object Tracking Buffer Endpoints
        @self.app.get("/api/buffer/objects")
        async def get_tracked_objects():
            """Get all currently tracked objects from buffer"""
            try:
                tracked_objects = self.object_buffer.get_all_tracked_objects()
                return {
                    "tracked_objects": tracked_objects,
                    "count": len(tracked_objects),
                    "last_updated": datetime.now().isoformat(),
                    "buffer_max_age_minutes": self.object_buffer.max_age_seconds / 60
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/buffer/object/{object_class}")
        async def get_object_last_position(object_class: str):
            """Get last known position of a specific object"""
            try:
                position_data = self.object_buffer.get_last_known_position(object_class)
                if position_data:
                    return {
                        "found": True,
                        "object_class": object_class,
                        "position_data": position_data,
                        "age_minutes": (time.time() - position_data['last_seen']) / 60
                    }
                else:
                    return {
                        "found": False,
                        "object_class": object_class,
                        "message": "Object not found in buffer or data too old"
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/buffer/navigate_to/{object_class}")
        async def navigate_to_last_position(object_class: str):
            """Get navigation instructions to last known position of object"""
            try:
                if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
                    raise HTTPException(status_code=400, detail="Camera not available for frame reference")
                
                # Get a frame for reference
                ret, frame = self.cap.read()
                if not ret:
                    raise HTTPException(status_code=400, detail="Could not capture frame for navigation")
                
                navigation_data = self.object_buffer.calculate_navigation_to_last_position(
                    object_class, frame.shape
                )
                
                if navigation_data:
                    return {
                        "success": True,
                        "object_class": object_class,
                        "navigation": navigation_data,
                        "instruction": navigation_data['instruction'],
                        "confidence": navigation_data['confidence'],
                        "last_seen_minutes_ago": navigation_data['last_seen_minutes_ago']
                    }
                else:
                    return {
                        "success": False,
                        "object_class": object_class,
                        "message": "No valid position data found for navigation"
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/buffer/clear")
        async def clear_buffer():
            """Clear all objects from tracking buffer"""
            try:
                self.object_buffer.tracked_objects.clear()
                self.object_buffer.save_buffer()
                return {
                    "success": True,
                    "message": "Object tracking buffer cleared"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/buffer/object/{object_class}")
        async def remove_object_from_buffer(object_class: str):
            """Remove specific object from tracking buffer"""
            try:
                if object_class in self.object_buffer.tracked_objects:
                    del self.object_buffer.tracked_objects[object_class]
                    self.object_buffer.save_buffer()
                    return {
                        "success": True,
                        "message": f"Removed {object_class} from buffer"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Object {object_class} not found in buffer"
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/buffer/toggle")
        async def toggle_object_tracking():
            """Toggle object tracking buffer on/off"""
            try:
                self.object_tracking_active = not self.object_tracking_active
                return {
                    "success": True,
                    "object_tracking_active": self.object_tracking_active,
                    "message": f"Object tracking {'enabled' if self.object_tracking_active else 'disabled'}"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/speak")
        async def speak(request: Request):
            """Generate TTS audio using Google Text-to-Speech with streaming response"""
            try:
                data = await request.json()
                text = data.get("text", "")
                lang = data.get("lang", "en")
                
                if not text.strip():
                    return JSONResponse(status_code=400, content={"error": "Missing text"})
                
                if not GTTS_AVAILABLE:
                    return JSONResponse(status_code=500, content={"error": "gTTS not available. Install with: pip install gtts"})
                
                # Support Nepali language like your code
                tts_lang = "ne" if lang == "ne" else "en"
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                
                # Stream the audio directly without saving to disk
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                
                return StreamingResponse(audio_fp, media_type="audio/mpeg")
                
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.post("/api/speak_current")
        async def speak_current():
            """Generate TTS audio for the current navigation instruction"""
            if not GTTS_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "gTTS not available. Install with: pip install gtts"})
            
            try:
                # Get current instruction
                current_text = self.current_instruction.natural_language
                if not current_text or not current_text.strip():
                    return JSONResponse(status_code=400, content={"error": "No current instruction available"})
                
                # Generate unique filename
                filename = f"current_{uuid.uuid4().hex}.mp3"
                path = os.path.join(self.audio_dir, filename)
                
                # Generate speech
                tts = gTTS(text=current_text, lang="en")
                tts.save(path)
                
                # Return audio file
                return FileResponse(path, media_type="audio/mpeg", filename=filename)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.get("/api/auto_tts_audio")
        async def get_auto_tts_audio():
            """Get the latest auto-generated TTS audio file"""
            if not GTTS_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "gTTS not available"})
            
            try:
                # Check if we have a latest auto audio file
                if hasattr(self, 'latest_auto_audio') and os.path.exists(self.latest_auto_audio):
                    return FileResponse(self.latest_auto_audio, media_type="audio/mpeg")
                else:
                    return JSONResponse(status_code=404, content={"error": "No auto TTS audio available"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "navigation": self.processing_active,
                    "camera": self.cap is not None and self.cap.isOpened() if self.cap else False,
                    "tts": self.tts_enabled,
                    "websocket_connections": len(self.active_connections)
                },
                "version": "1.0.0"
            }
        
        @self.app.get("/api/stats")
        async def get_navigation_stats():
            """Get navigation statistics"""
            try:
                stats = {
                    "current_target": self.navigation_system.target_item,
                    "total_frames_processed": getattr(self.navigation_system, 'frame_count', 0),
                    "active_connections": len(self.active_connections),
                    "tts_enabled": self.tts_enabled,
                    "last_instruction": self.current_instruction.dict() if hasattr(self.current_instruction, 'dict') else self.current_instruction.__dict__,
                    "processing_active": self.processing_active,
                    "camera_resolution": {
                        "width": 1920 if self.cap else 0,
                        "height": 1080 if self.cap else 0
                    }
                }
                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/favicon.ico")
        async def favicon():
            """Serve favicon to prevent 404 errors"""
            return JSONResponse(status_code=204, content={})
        
        @self.app.get("/api/audio/{filename}")
        async def serve_audio(filename: str):
            """Serve audio files"""
            try:
                audio_path = os.path.join(self.audio_dir, filename)
                if os.path.exists(audio_path):
                    return FileResponse(audio_path, media_type="audio/mpeg")
                else:
                    raise HTTPException(status_code=404, detail="Audio file not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/translate")
        async def translate_to_en(request: Request):
            """Translate text to English using Google Translate"""
            try:
                data = await request.json()
                text = data.get("text", "")
                
                if not text.strip():
                    return JSONResponse({"translated": ""})
                
                if not self.translate_enabled:
                    return JSONResponse(status_code=500, content={"error": "Google Translate not available. Install with: pip install googletrans==4.0.0-rc1"})
                
                translated = self.translator.translate(text, dest="en")
                return JSONResponse({"translated": translated.text})
                
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.post("/api/find")
        async def find_objects(request: Request):
            """Respond with found objects in specified language"""
            try:
                data = await request.json()
                objects = data.get("objects", [])
                lang = data.get("lang", "en")
                
                if not objects:
                    return JSONResponse({"response": "No objects provided."})
                
                if lang == "ne":
                    translated_objects = ", ".join(objects)
                    response_text = f"मैले {translated_objects} भेटेँ।"
                else:
                    response_text = f"I found {', '.join(objects)}."
                
                return JSONResponse({"response": response_text})
                
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.post("/api/no-object")
        async def no_object(request: Request):
            """Respond when no objects are found in specified language"""
            try:
                data = await request.json()
                lang = data.get("lang", "en")
                
                if lang == "ne":
                    response_text = "मैले चिनिएको कुनै वस्तु भेटिन।"
                else:
                    response_text = "I didn't detect any known object."
                
                return JSONResponse({"response": response_text})
                
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
    
    def convert_to_natural_language(self, technical_instruction: str, navigation_data: Dict = None) -> str:
        """Convert technical instructions to natural, conversational language using consistent patterns from main system"""
        
        if not technical_instruction:
            return "I'm looking around to help you."
        
        # Handle memory guidance (when using last known position)
        if "MEMORY_GUIDANCE:" in technical_instruction:
            # Extract the guidance instruction and timing info
            parts = technical_instruction.split("MEMORY_GUIDANCE:", 1)
            if len(parts) > 1:
                guidance_part = parts[1].strip()
                # Extract the instruction and timing
                if "(" in guidance_part and "min ago)" in guidance_part:
                    instruction_part = guidance_part.split("(")[0].strip()
                    time_part = guidance_part.split("(")[1].replace(")", "").strip()
                    
                    # Create natural language for memory guidance
                    if "turn right" in instruction_part.lower():
                        direction = "turn to your right"
                    elif "turn left" in instruction_part.lower():
                        direction = "turn to your left"
                    elif "look up" in instruction_part.lower():
                        direction = "look up"
                    elif "look down" in instruction_part.lower():
                        direction = "look down"
                    else:
                        direction = instruction_part.lower()
                    
                    return f"I remember seeing the {self.navigation_system.target_item} nearby. Try to {direction} - I saw it there {time_part}."
                else:
                    return f"I remember seeing the {self.navigation_system.target_item} in this area. Let me guide you to where I last saw it."
        
        # Remove technical markers
        instruction = technical_instruction.replace("[", "").replace("]", "")
        
        # Handle success cases
        if "SUCCESS" in instruction or "TARGET REACHED" in instruction:
            return "Perfect! You've reached your target."
        
        # Handle search/initialization states
        if "SEARCH" in instruction or "Searching" in instruction:
            if "person" in instruction.lower():
                return "I'm looking for you in the camera. Please step into view."
            else:
                return f"I can see you! Now searching for the {self.navigation_system.target_item}."
        
        if "Initializing" in instruction or "Processing" in instruction:
            return "Getting ready to help you navigate."
        
        # Extract key direction information from the technical instruction
        natural_parts = []
        
        # Handle very close positioning (fine adjustments)
        if "VERY CLOSE" in instruction or "HOT" in instruction:
            if "right" in instruction.lower():
                natural_parts.append("Adjust slightly to your right.")
            elif "left" in instruction.lower():
                natural_parts.append("Adjust slightly to your left.")
            elif "forward" in instruction.lower():
                natural_parts.append("Move forward just a bit.")
            elif "back" in instruction.lower():
                natural_parts.append("Step back a little.")
            else:
                natural_parts.append("You're very close! Make small adjustments.")
            return " ".join(natural_parts)
        
        # Handle primary movement directions using natural patterns from main system
        direction_found = False
        
        # Check for turn around command
        if "Turn around" in instruction or "behind you" in instruction:
            natural_parts.append("Turn around.")
            direction_found = True
        
        # Check for forward movement
        elif any(phrase in instruction for phrase in ["Move FORWARD", "Move forward", "target is ahead", "straight ahead"]):
            if "slowly" in instruction:
                natural_parts.append("Walk forward slowly.")
            else:
                natural_parts.append("Walk forward.")
            direction_found = True
        
        # Check for turning instructions
        elif "Turn RIGHT" in instruction or "to your right" in instruction:
            if "slightly" in instruction or "Bear slightly" in instruction:
                natural_parts.append("Turn slightly to your right.")
            else:
                natural_parts.append("Turn to your right.")
            direction_found = True
            
        elif "Turn LEFT" in instruction or "to your left" in instruction:
            if "slightly" in instruction or "Bear slightly" in instruction:
                natural_parts.append("Turn slightly to your left.")
            else:
                natural_parts.append("Turn to your left.")
            direction_found = True
        
        # Handle "OK" or "Facing target" cases - always provide forward guidance
        if ("OK" in instruction or "Facing target" in instruction) and not direction_found:
            natural_parts.append("Walk forward.")
            direction_found = True
            
            # Check for additional side adjustments
            if "RIGHT" in instruction or "right" in instruction.lower():
                natural_parts.append("Bear slightly to your right.")
            elif "LEFT" in instruction or "left" in instruction.lower():
                natural_parts.append("Bear slightly to your left.")
        
        # Check for combined directions (ahead and to your right/left)
        if "ahead and to your right" in instruction:
            natural_parts = ["Walk forward and turn slightly to your right."]
            direction_found = True
        elif "ahead and to your left" in instruction:
            natural_parts = ["Walk forward and turn slightly to your left."]
            direction_found = True
        
        # Handle backward movement
        if "Step back" in instruction or "Move backward" in instruction:
            natural_parts.append("Step backward.")
            direction_found = True
        
        # Vertical guidance with natural language
        if "Look UP" in instruction:
            natural_parts.append("Look up.")
        elif "Look DOWN" in instruction:
            natural_parts.append("Look down.")
        
        # If we found directions, return them
        if natural_parts:
            return " ".join(natural_parts)
        
        # Fallback analysis for any missed cases
        if not direction_found:
            # Parse remaining cases and provide natural language
            clean_instruction = instruction.lower()
            
            if "right" in clean_instruction:
                if "forward" in clean_instruction:
                    return "Walk forward and turn slightly to your right."
                else:
                    return "Turn to your right."
            elif "left" in clean_instruction:
                if "forward" in clean_instruction:
                    return "Walk forward and turn slightly to your left."
                else:
                    return "Turn to your left."
            elif "forward" in clean_instruction or "ahead" in clean_instruction:
                return "Walk forward."
            elif "back" in clean_instruction:
                return "Step backward."
            elif "around" in clean_instruction:
                return "Turn around."
            else:
                return "Keep moving forward."
        
        return "Keep moving forward."
    
    def stabilize_instruction(self, new_instruction: str) -> str:
        """Stabilize instructions using mode of recent frames to reduce high FPS sensitivity"""
        
        # Add new instruction to history
        self.instruction_history.append(new_instruction)
        
        # Keep only last N instructions
        if len(self.instruction_history) > self.max_history_size:
            self.instruction_history.pop(0)
        
        # Need at least 3 instructions to stabilize
        if len(self.instruction_history) < 3:
            return new_instruction
        
        # Extract core direction from each instruction for comparison
        def extract_core_direction(instruction):
            """Extract the main direction/action from instruction"""
            instruction_lower = instruction.lower()
            
            if "turn around" in instruction_lower:
                return "turn_around"
            elif "turn right" in instruction_lower or "adjust right" in instruction_lower:
                return "turn_right"
            elif "turn left" in instruction_lower or "adjust left" in instruction_lower:
                return "turn_left"
            elif "move forward" in instruction_lower or "step forward" in instruction_lower:
                return "move_forward"
            elif "step back" in instruction_lower:
                return "move_back"
            elif "very close" in instruction_lower or "target is very close" in instruction_lower:
                return "very_close"
            elif "close" in instruction_lower:
                return "close"
            elif "far" in instruction_lower:
                return "far"
            elif "good direction" in instruction_lower or "facing" in instruction_lower:
                return "good_direction"
            elif "look up" in instruction_lower:
                return "look_up"
            elif "look down" in instruction_lower:
                return "look_down"
            elif "reached" in instruction_lower or "perfect" in instruction_lower:
                return "reached"
            elif "searching" in instruction_lower or "looking" in instruction_lower:
                return "searching"
            else:
                return "other"
        
        # Get core directions for all recent instructions
        core_directions = [extract_core_direction(inst) for inst in self.instruction_history]
        
        # Find the most common direction (mode)
        from collections import Counter
        direction_counts = Counter(core_directions)
        most_common_direction = direction_counts.most_common(1)[0][0]
        
        # If the most common direction appears in at least 60% of recent frames, use it
        stability_threshold = max(2, len(self.instruction_history) * 0.6)
        
        if direction_counts[most_common_direction] >= stability_threshold:
            # Find the most recent instruction with this direction
            for instruction in reversed(self.instruction_history):
                if extract_core_direction(instruction) == most_common_direction:
                    self.last_stable_instruction = instruction
                    return instruction
        
        # If no stable direction, return the new instruction but don't change too rapidly
        current_direction = extract_core_direction(new_instruction)
        if self.last_stable_instruction:
            last_direction = extract_core_direction(self.last_stable_instruction)
            
            # If direction changed dramatically, require more confirmation
            if current_direction != last_direction and direction_counts[current_direction] < 2:
                return self.last_stable_instruction
        
        self.last_stable_instruction = new_instruction
        return new_instruction
    
    def extract_navigation_data(self, technical_instruction: str) -> Dict[str, Any]:
        """Extract structured data from technical instructions"""
        data = {
            'distance_to_target': None,
            'facing_target': False,
            'target_direction': 'unknown',
            'confidence': 0.5
        }
        
        # Extract distance
        if "cm away" in technical_instruction:
            try:
                distance_str = technical_instruction.split("cm away")[0].split()[-1]
                data['distance_to_target'] = float(distance_str)
            except:
                pass
        elif "m away" in technical_instruction:
            try:
                distance_str = technical_instruction.split("m away")[0].split()[-1]
                data['distance_to_target'] = float(distance_str) * 100  # Convert to cm
            except:
                pass
        
        # Extract facing status
        data['facing_target'] = "Facing target" in technical_instruction or "[OK]" in technical_instruction
        
        # Extract direction
        if "behind you" in technical_instruction:
            data['target_direction'] = "behind"
        elif "ahead" in technical_instruction or "forward" in technical_instruction:
            data['target_direction'] = "ahead"
        elif "right" in technical_instruction.lower():
            data['target_direction'] = "right"
        elif "left" in technical_instruction.lower():
            data['target_direction'] = "left"
        
        # Set confidence based on instruction type
        if "SUCCESS" in technical_instruction or "HOT" in technical_instruction:
            data['confidence'] = 0.9
        elif "SEARCH" in technical_instruction:
            data['confidence'] = 0.3
        else:
            data['confidence'] = 0.7
        
        return data
    
    async def start_camera_processing(self):
        """Start camera capture and processing threads"""
        if self.processing_active:
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_id}")
        
        # Set camera properties for 1080p resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 1080p width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 1080p height
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.processing_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"Navigation server started - Target: {self.navigation_system.target_item}")
    
    async def stop_camera_processing(self):
        """Stop camera capture and processing"""
        self.processing_active = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Navigation server stopped")
    
    def processing_loop(self):
        """Main processing loop for camera frames"""
        last_instruction_time = 0
        instruction_cooldown = 5.0  # Give instructions every 5 seconds
        last_tts_time = 0
        tts_cooldown = 5.0  # Speak every 5 seconds
        last_cleanup_time = 0
        cleanup_interval = 60.0  # Clean up every minute
        
        while self.processing_active:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Process frame with navigation system
                processed_frame = self.navigation_system.process_frame(frame.copy())
                
                # Update object tracking buffer every few seconds
                if (self.object_tracking_active and 
                    current_time - self.last_buffer_update >= self.buffer_update_interval):
                    self.update_object_buffer(frame)
                    self.last_buffer_update = current_time
                
                # Update frame queue (non-blocking)
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame, block=False)
                except:
                    pass
                
                # Generate instructions every 5 seconds regardless of changes
                current_time = time.time()
                if current_time - last_instruction_time >= instruction_cooldown:
                    
                    # Get current instruction from navigation system
                    technical_instruction = getattr(self.navigation_system, '_last_instruction', 'Processing...')
                    
                    # Check detection status
                    person_detected = self.navigation_system.person_tracker.get_best_person()[0] is not None
                    target_detected = self.navigation_system.target_tracker.get_best_target()[0] is not None
                    
                    # If target is not detected, try to guide to last known position
                    if not target_detected:
                        buffer_navigation = self.object_buffer.calculate_navigation_to_last_position(
                            self.navigation_system.target_item, 
                            frame.shape
                        )
                        if buffer_navigation:
                            technical_instruction = f"MEMORY_GUIDANCE: {buffer_navigation['instruction']} (last seen {buffer_navigation['last_seen_minutes_ago']:.1f} min ago)"
                            print(f"🧠 Using memory guidance: {technical_instruction}")
                    
                    # Convert to natural language
                    natural_instruction_raw = self.convert_to_natural_language(technical_instruction)
                    
                    # Apply stabilization to reduce high FPS sensitivity
                    natural_instruction = self.stabilize_instruction(natural_instruction_raw)
                    
                    # Extract navigation data
                    nav_data = self.extract_navigation_data(technical_instruction)
                    
                    # Create instruction object
                    instruction = NavigationInstruction(
                        instruction=technical_instruction,
                        natural_language=natural_instruction,
                        confidence=nav_data['confidence'],
                        timestamp=datetime.now().isoformat(),
                        person_detected=person_detected,
                        target_detected=target_detected,
                        distance_to_target=nav_data['distance_to_target'],
                        facing_target=nav_data['facing_target'],
                        target_direction=nav_data['target_direction']
                    )
                    
                    self.current_instruction = instruction
                    
                    # Add to instruction queue for WebSocket
                    try:
                        if not self.instruction_queue.full():
                            self.instruction_queue.put(instruction, block=False)
                    except:
                        pass
                    
                    last_instruction_time = current_time
                    
                    # Broadcast to WebSocket clients
                    if self.active_connections:
                        asyncio.create_task(self.broadcast_instruction(instruction))
                
                # TTS output - speak every 5 seconds independently of instruction generation using Google TTS
                if (self.tts_enabled and 
                    current_time - last_tts_time >= tts_cooldown):
                    try:
                        # Always speak the current natural instruction every 5 seconds
                        current_natural_instruction = self.current_instruction.natural_language
                        if (current_natural_instruction and 
                            current_natural_instruction.strip() and
                            "Initializing" not in current_natural_instruction and
                            "Getting ready" not in current_natural_instruction):
                            
                            print(f"Generating TTS audio at {current_time:.1f}s: {current_natural_instruction}")
                            
                            # Generate Google TTS audio file
                            audio_success = self.generate_tts_audio(current_natural_instruction)
                            
                            if audio_success:
                                # Broadcast TTS notification to all connected clients
                                asyncio.create_task(self.broadcast_tts_instruction(current_natural_instruction))
                                self.last_spoken_instruction = current_natural_instruction
                                last_tts_time = current_time
                                print(f"TTS audio generated and broadcast - next speech at {last_tts_time + tts_cooldown:.1f}s")
                            else:
                                print("Failed to generate TTS audio")
                                
                    except Exception as e:
                        print(f"TTS generation error: {e}")
                
                # Cleanup old audio files periodically
                if current_time - last_cleanup_time >= cleanup_interval:
                    self.cleanup_old_audio_files()
                    # Also cleanup old buffer entries
                    self.object_buffer.cleanup_old_entries()
                    last_cleanup_time = current_time
                
                time.sleep(0.04)  # Slightly slower FPS for better stability (~25 FPS)
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def update_object_buffer(self, frame):
        """
        Update the object tracking buffer with current detections
        """
        try:
            # Get current detections from the navigation system
            if hasattr(self.navigation_system, 'model'):
                # Run YOLO detection on current frame
                results = self.navigation_system.model(frame)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = self.navigation_system.model.names[cls]
                            confidence = float(box.conf[0])
                            bbox = box.xyxy[0].cpu().numpy()
                            
                            # Only track objects with reasonable confidence
                            if confidence > 0.3:
                                # Estimate depth if available
                                depth_estimate = None
                                if hasattr(self.navigation_system, 'estimate_depth_multi_method'):
                                    depth_estimate = self.navigation_system.estimate_depth_multi_method(
                                        bbox, class_name, frame_shape=frame.shape
                                    )
                                
                                # Update buffer
                                self.object_buffer.update_object_position(
                                    object_class=class_name,
                                    bbox=tuple(bbox),
                                    confidence=confidence,
                                    frame_shape=frame.shape,
                                    depth_estimate=depth_estimate
                                )
                
                # Save buffer periodically
                self.object_buffer.save_buffer()
                
        except Exception as e:
            print(f"Error updating object buffer: {e}")
    
    def generate_video_stream(self):
        """Generate video stream for HTTP endpoint"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Empty:
                # No frame available, send a black frame in 1080p
                black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    async def websocket_handler(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                try:
                    instruction = self.instruction_queue.get_nowait()
                    await websocket.send_text(instruction.json())
                except Empty:
                    # Send current status
                    await websocket.send_text(self.current_instruction.json())
                
                await asyncio.sleep(0.5)  # Update every 500ms
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast_instruction(self, instruction: NavigationInstruction):
        """Broadcast instruction to all WebSocket clients"""
        if not self.active_connections:
            return
        
        message = instruction.json()
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections.remove(websocket)
    
    async def broadcast_tts_instruction(self, text: str):
        """Broadcast TTS instruction to all WebSocket clients for automatic playback"""
        if not self.active_connections:
            return
        
        # Create a special TTS message
        tts_message = {
            "type": "auto_tts",
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        
        message = json.dumps(tts_message)
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    def generate_tts_audio(self, text: str, lang: str = "en") -> bool:
        """Generate Google TTS audio file and return success status"""
        try:
            if not GTTS_AVAILABLE:
                return False
            
            # Generate unique filename for the audio
            filename = f"auto_{uuid.uuid4().hex}.mp3"
            path = os.path.join(self.audio_dir, filename)
            
            # Generate speech using Google TTS
            tts = gTTS(text=text, lang=lang)
            tts.save(path)
            
            # Store the latest auto-generated audio file path for cleanup
            self.latest_auto_audio = path
            
            print(f"Generated TTS audio: {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating TTS audio: {e}")
            return False
    
    def cleanup_old_audio_files(self):
        """Clean up old audio files to prevent disk space issues"""
        try:
            if not os.path.exists(self.audio_dir):
                return
            
            current_time = time.time()
            for filename in os.listdir(self.audio_dir):
                if filename.endswith('.mp3'):
                    file_path = os.path.join(self.audio_dir, filename)
                    # Delete files older than 5 minutes
                    if os.path.getctime(file_path) < current_time - 300:
                        try:
                            os.remove(file_path)
                            print(f"Cleaned up old audio file: {filename}")
                        except OSError:
                            pass
        except Exception as e:
            print(f"Error cleaning up audio files: {e}")
    
    # ...existing code...

# Server instance
server = None

def create_server(target_item: str = 'bottle', camera_id: int = 0, host: str = "127.0.0.1", port: int = 8000):
    """Create and configure the navigation server"""
    global server
    server = NavigationServer(target_item=target_item, camera_id=camera_id)
    return server.app

def run_server(target_item: str = 'bottle', camera_id: int = 0, host: str = "127.0.0.1", port: int = 8000):
    """Run the navigation server"""
    print(f"""
🧭 Navigation Assistant Server
==============================
Target Item: {target_item}
Camera ID: {camera_id}
Server: http://{host}:{port}
API Docs: http://{host}:{port}/docs

Features:
- Real-time navigation with natural language instructions
- Web interface for monitoring
- WebSocket for live updates
- Text-to-speech output with multilingual support
- Translation API (English/Nepali)
- Object detection responses
- Video streaming
- REST API for control

Press Ctrl+C to stop the server
    """)
    
    app = create_server(target_item, camera_id, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation Assistant Server")
    parser.add_argument("--target", default="bottle", help="Target item to track")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    
    args = parser.parse_args()
    
    try:
        run_server(
            target_item=args.target,
            camera_id=args.camera,
            host=args.host,
            port=args.port
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
