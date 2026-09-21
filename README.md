# 🛡️ AI-Based Intelligent Video Analytics Platform for Border Surveillance

### Smart India Hackathon 2026 | PS ID: SIH26187

An AI-powered surveillance platform that transforms existing CCTV infrastructure into an intelligent border monitoring system using Computer Vision and AI.

## 🚀 Key Features

- 👤 Real-time Human Detection & Tracking
- 🚗 Vehicle Detection & Classification
- 🧑 Face Detection & Recognition
- 🔢 Automatic Number Plate Recognition (ANPR)
- 📍 Geofencing & Virtual Fence Detection
- 📹 Multi-Camera Surveillance
- 🚨 Real-Time Event & Alert Monitoring
- 🖥️ Centralized Surveillance Dashboard
- 🌐 Remote Camera Connectivity

## 🧠 Technology Stack

**Frontend:** React.js, JavaScript, Vite  
**Backend:** Python, aiohttp, WebRTC, OpenCV  
**AI/ML:** YOLO, YuNet, SFace, ByteTrack, MediaPipe  
**Database:** SQLite / MongoDB  
**Deployment:** Vercel, Cloudflare Tunnel  
**Version Control:** Git, GitHub

## 🏗️ System Flow

```text
CCTV / Webcam / Mobile Camera
            ↓
      Video Processing
            ↓
     YOLO Detection
            ↓
     Object Tracking
       ↙          ↘
Face Detection   Vehicle Analysis
   ↓                  ↓
YuNet + SFace       ANPR
       ↘            ↙
       Event Detection
            ↓
   Centralized Dashboard
