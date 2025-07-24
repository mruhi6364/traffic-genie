# 🚦 Smart Traffic Management System

A Flask-based web application for intelligent traffic video analysis and visualization. The system simulates object detection using a YOLO-style model (MobileNet SSD), processes video frames to detect vehicles, and provides 3D-like SVG visualizations and interactive stats.

---


## 🔧 Features

- 🎥 Upload traffic videos (MP4 format)
- 🚘 Vehicle detection using simulated YOLO (MobileNet SSD via OpenCV DNN)
- 📊 Frame-wise detection statistics and metadata
- 🧠 Session-based storage and simulated tracking
- 🖼️ 3D-style SVG rendering of traffic frames
- 📈 Chart.js integration for live traffic graphs and analytics

---

## 📁 Folder Structure

```
smart-traffic-system/
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
├── app.py
├── detector.py
├── utils.py
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-traffic-system.git
cd smart-traffic-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required Libraries:
- Flask
- OpenCV (`opencv-python`)
- NumPy
- Chart.js (included via CDN)
- SVG.js (optional, included in template)

### 3. Run the App

```bash
python main.py
```

Visit: `http://127.0.0.1:5000` in your browser.

---

## 📦 How it Works

1. **Upload** a traffic video from the homepage.
2. The backend splits the video into frames using OpenCV.
3. Each frame is passed to a YOLO-style detector (MobileNet SSD).
4. Detected vehicles are classified as `car`, `truck`, `bike`, or `bus`.
5. SVG overlays simulate a 3D visual representation of traffic.
6. Detection data is plotted using Chart.js for better understanding of density and flow.

---

## 🛠️ TODO / Future Improvements

- ✅ Real-time camera feed support
- ✅ Add DeepSORT or ByteTrack for ID tracking
- 🔲 YOLOv8 Integration (PyTorch/ONNX)
- 🔲 Heatmap overlays for traffic density
- 🔲 Export analytics report as PDF

---
## Screenshots

![1](https://github.com/user-attachments/assets/5d275dd4-5ee7-4010-8746-da4398e19e25)

![2](https://github.com/user-attachments/assets/1a700b07-c90b-42a8-9861-bac7e6f441dd)

---

## 👥 Team Members

### 🔧 ML & YOLO Implementation Team:
- [Parth Bharadia](https://github.com/ParthBharadia)
- [Debarpita Dutta](https://github.com/devv712)
- [Ruhani ](https://github.com/mruhi6364)
- [Akshun Mehrotra](https://github.com/Akshunmehrotra57)

### 🌐 Web Development Team:
- [Vatsal Unadkat](https://github.com/Vatsaalll)
- [Shambhavi Choubey](https://github.com/ShambhaviChoubey)
- [Rim Patel](https://github.com/Rimpatel14)
- [Vishwam Patel](https://github.com/VishvmPatel)
---

