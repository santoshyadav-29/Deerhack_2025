from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO
import speech_recognition as sr
import torch

app = FastAPI()

# Initialize YOLO model with GPU support if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")
model = YOLO('yolo11l.pt')
model.to(device)
target_class_name = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def get_target_class_name():
    """Get target class name using speech recognition"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Please say the target item (e.g., 'cell phone', 'person', 'bottle'):")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

@app.post('/api/set_target')
async def set_target(request: Request):
    """Set target object for detection via API"""
    global target_class_name
    data = await request.json()
    target_class_name = data.get('target')
    print(f"Target set to: {target_class_name}")
    return JSONResponse({'status': 'ok', 'target': target_class_name})

@app.get('/api/get_target')
async def get_target():
    """Get current target object"""
    return JSONResponse({'target': target_class_name})

@app.get('/api/available_classes')
async def get_available_classes():
    """Get list of available YOLO classes"""
    return JSONResponse({'classes': list(model.names.values())})

def process_frame_with_detection(frame):
    """Process frame with YOLO object detection"""
    global target_class_name
    
    if target_class_name is None:
        return frame, False, []
    
    # Run YOLO detection
    results = model(frame)
    detected = False
    detected_objects = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                confidence = float(box.conf[0])
                
                # Store all detected objects
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence
                })
                
                # Check if it's our target class
                if class_name == target_class_name:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box in green for target
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    # Draw other objects in blue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, f"{class_name}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    
    return frame, detected, detected_objects

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    global target_class_name
    await websocket.accept()
    print("🔌 WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle different message types
            if data.startswith("data:image"):
                # Extract base64 content
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                np_img = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Process frame with YOLO detection
                    processed_frame, target_detected, detected_objects = process_frame_with_detection(frame)
                    
                    # Display frame using OpenCV
                    cv2.imshow("Object Detection Feed", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # Send detection results back to client
                    result = {
                        "type": "detection_result",
                        "target_detected": target_detected,
                        "target": target_class_name,
                        "detected_objects": detected_objects,
                        "total_objects": len(detected_objects)
                    }
                    
                    await websocket.send_text(json.dumps(result))
            
            elif data.startswith("{"):
                # Handle JSON messages (like setting target)
                try:
                    message = json.loads(data)
                    if message.get("type") == "set_target":
                        target_class_name = message.get("target")
                        response = {
                            "type": "target_set",
                            "target": target_class_name,
                            "status": "success"
                        }
                        await websocket.send_text(json.dumps(response))
                        print(f"Target set via WebSocket: {target_class_name}")
                    
                    elif message.get("type") == "get_classes":
                        response = {
                            "type": "available_classes",
                            "classes": list(model.names.values())
                        }
                        await websocket.send_text(json.dumps(response))
                        
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    finally:
        cv2.destroyAllWindows()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("🚀 Server starting up...")
    print(f"📹 YOLO model loaded successfully on {device}")
    print("🎯 Available object classes:", len(model.names))
    print("💡 You can set target via API or speech recognition")
    if torch.cuda.is_available():
        print(f"🔥 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU - consider using GPU for better performance")
    
    # Optionally get target via speech recognition on startup
    # Uncomment the following lines if you want to use speech recognition on startup
    # global target_class_name
    # target_class_name = get_target_class_name()
    # if target_class_name:
    #     print(f"🎯 Target set to: {target_class_name}")

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting Object Detection WebSocket Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
