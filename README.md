# TomatoIQ — Offline Arm Tomato Leaf Disease Classification & Farmer Advisory

TomatoIQ is an offline-first tomato leaf disease diagnostic and multilingual farmer advisory system built for Arm edge platforms (Raspberry Pi 5 / Apple Silicon macOS) for the Bharat AI-SoC Student Challenge.

## Key Features
- **100% Offline Pipeline (0 KB Cloud Inference Payload):** Combines leaf and lesion isolation (`extract_main_leaf_from_camera` / `isolate_leaf_and_lesions`), INT8-quantized ResNet50 classification (`models/tomato_resnet50_int8.tflite`), a 10%-severity-bracket KVK treatment database, and local Llama 3.2 farmer guidance.
- **4-Class Detection (99.01% Validation Accuracy):** Early Blight, Late Blight, Yellow Leaf Curl Virus (TYLCV), and Healthy Tomato Crop.
- **Multilingual Support & Offline TTS:** English, Hindi, and Kannada diagnostics, dosage tables, follow-up farmer chat (`/chat`), and offline audio advisory (`/generate_voice`).

## Run Locally
```bash
python app.py
