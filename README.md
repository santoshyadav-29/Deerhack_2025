# Deerhack2025
# Keyo 🗝️  
**Voice-Activated Indoor Navigation for the Visually Impaired**

Keyo is an assistive technology project that enables visually impaired individuals to navigate indoor environments using real-time object detection, voice guidance, and natural language interaction.  
The system scans a room via webcam or CCTV, logs object positions, and verbally guides users to requested items like keys — all through voice commands.

---

## 🚀 Features
- **Voice-Activated Navigation** – Hands-free interaction for ease of use.
- **Real-Time Object Detection** – Detects and tracks objects via OpenCV and MediaPipe.
- **Last Known Location Memory** – Remembers where objects were last seen.
- **Natural Language Interaction** – Users can ask questions like “Where are my keys?”.
- **Object Description Mode** – Provides verbal descriptions of surroundings on request.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries & Frameworks:** OpenCV, MediaPipe, Pyttsx3 (Text-to-Speech), SpeechRecognition  
- **Hardware:** Standard webcam or CCTV feed  
- **Platform:** Desktop (Windows/Linux/Mac)  

---

## 📸 How It Works
1. **Initialization** – System starts listening for commands.  
2. **Scanning** – Webcam or CCTV scans the environment in real time.  
3. **Detection** – Objects are identified and their positions logged.  
4. **Voice Query** – User asks for an item’s location.  
5. **Guidance** – System verbally guides the user to the item.  

---

## 🖥️ Installation & Setup
```bash
# Clone the repository

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
