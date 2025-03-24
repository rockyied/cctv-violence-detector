# 🛡️ CCTV Violence Detector (Binary Classification)

This project detects violence from CCTV video frames using a fine-tuned MobileNetV2 model. It classifies footage into **Normal** or **Violence** activity.

---

## 📁 Dataset

We use the [Smart-City CCTV Violence Detection (SCVD)](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd/data) dataset from Kaggle.

- The dataset contains `.avi` videos labeled as:
  - `Normal`
  - `Violence`
  - `Weapons`

---

## Preprocessing

1. Download videos from Kaggle.
2. Convert videos to frames (5 frames/video):
```python
import cv2, os
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 5)
    for i in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), frame)


Organize the dataset like this:

css
Copy
Edit
frames/
├── train/
│   ├── Normal/
│   └── Violence/
└── test/
    ├── Normal/
    └── Violence


🧠 Model
Model Used: MobileNetV2

Framework: TensorFlow / Keras

Final Accuracy: 100% on test data (binary classification)

🚀 App
An interactive Streamlit app is included:

Upload a zip of 5 frames (0.jpg to 4.jpg)

Get predictions with confidence scores

🧾 Author
Made with ❤️ by Rakesh (Marquette University)

