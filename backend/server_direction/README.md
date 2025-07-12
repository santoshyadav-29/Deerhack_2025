# 🧭 Navigation Assistant

A real-time object navigation system with voice guidance and web interface.

## ✨ Features

- **Real-time Navigation**: Navigate to objects using computer vision
- **Voice Guidance**: Automatic text-to-speech instructions every 5 seconds
- **Modern Web Interface**: Beautiful, responsive UI that works on mobile devices
- **Live Video Feed**: Real-time camera stream with object detection overlays
- **Multiple Target Objects**: Support for 80+ different object types
- **WebSocket Updates**: Real-time status and instruction updates
- **Google TTS**: High-quality text-to-speech using Google TTS

## 📋 Requirements

- Python 3.8+
- Webcam or camera device
- Speakers or headphones for audio output
- Internet connection (for Google TTS)

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn opencv-python gtts pydantic
   ```

2. **Run the Server**:
   ```bash
   python run_server.py
   ```

3. **Open Web Interface**:
   - Navigate to: http://localhost:8000
   - On mobile: http://[your-ip]:8000

## 🎮 Usage

### Web Interface

1. **Start Navigation**: Click "Start" button to begin camera processing
2. **Select Target**: Choose an object from the dropdown menu
3. **Change Target**: Click "Change Target" to switch objects
4. **Voice Instructions**: Automatic voice guidance every 5 seconds
5. **Manual TTS**: Use "Speak Current" or "Test TTS" buttons

### Controls

- **🟢 Start**: Begin navigation and camera processing
- **🔴 Stop**: Stop navigation system
- **🔄 Reset**: Reset tracking data
- **🎯 Change Target**: Switch to a different object
- **🔊 Speak Current**: Manually play current instruction
- **🎵 Test TTS**: Test the TTS system

### Status Indicators

- **Green**: System active/detected
- **Red**: System inactive/not detected
- **Orange**: Tracking/searching state

## 📱 Mobile Support

The interface is fully responsive and optimized for:
- Smartphones (iOS/Android)
- Tablets
- Desktop browsers
- Touch and mouse interactions

## 🔧 API Endpoints

### Core Navigation
- `GET /` - Web interface
- `POST /api/start` - Start navigation
- `POST /api/stop` - Stop navigation
- `POST /api/reset` - Reset tracking
- `GET /api/status` - System status

### Target Management
- `GET /api/targets` - Available objects
- `POST /api/change_target/{item}` - Change target object

### Audio & TTS
- `POST /api/speak` - Generate TTS audio
- `POST /api/speak_current` - Speak current instruction
- `GET /api/auto_tts_audio` - Get auto-generated audio

### Data & Stats
- `GET /api/instruction` - Current instruction
- `GET /api/health` - Health check
- `GET /api/stats` - Navigation statistics
- `GET /api/video_feed` - Live video stream
- `WebSocket /ws` - Real-time updates

## 🎯 Supported Objects

The system can navigate to 80+ object types including:

**Electronics**: laptop, cell phone, tv, mouse, keyboard
**Furniture**: chair, couch, bed, dining table
**Kitchen**: bottle, cup, wine glass, utensils, appliances
**Personal**: backpack, handbag, book, clock
**And many more...**

## 🔊 Audio Features

- **Automatic TTS**: Voice instructions every 5 seconds during navigation
- **Google TTS**: High-quality speech synthesis
- **Web Audio**: Browser-based audio playback
- **Manual Controls**: On-demand speech generation
- **Mobile Compatible**: Works on all devices with speakers

## 🛠️ Configuration

### Camera Settings
- Default: Camera ID 0 (first camera)
- Resolution: 1080p (1920x1080)
- FPS: 30 frames per second

### Network Settings
- Default Host: 0.0.0.0 (accessible from network)
- Default Port: 8000
- WebSocket: Real-time communication

### TTS Settings
- Engine: Google Text-to-Speech (gTTS)
- Language: English (en)
- Interval: 5 seconds
- Volume: 80%

## 🐛 Troubleshooting

### Audio Not Playing
1. Interact with the webpage first (click/tap anywhere)
2. Check browser audio permissions
3. Ensure speakers/headphones are connected
4. Verify internet connection for Google TTS

### Camera Not Working
1. Check camera permissions in browser
2. Ensure camera is not used by other applications
3. Try different camera ID (0, 1, 2...)
4. Restart the server

### Connection Issues
1. Check firewall settings
2. Ensure port 8000 is available
3. Try different port: `python navigation_server.py --port 8080`
4. Check network connectivity

## 📱 Mobile Usage Tips

- **Portrait Mode**: Optimized layout for phone screens
- **Touch Controls**: Large, touch-friendly buttons
- **Auto-rotate**: Interface adapts to orientation
- **Offline**: Video and controls work without internet (TTS requires connection)

## 🔒 Security Notes

- Server binds to 0.0.0.0 for network access
- In production, configure proper firewall rules
- Consider HTTPS for public deployments
- Audio files are automatically cleaned up

## 📊 Performance

- **Low Latency**: Real-time video processing (~25 FPS)
- **Efficient**: Optimized for continuous operation
- **Scalable**: Multiple WebSocket connections supported
- **Resource Management**: Automatic cleanup of temporary files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. See LICENSE file for details.

---

**Made with ❤️ for accessible navigation**
