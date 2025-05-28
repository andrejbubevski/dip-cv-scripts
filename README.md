# 🧠 Face Detection and Manipulation Suite

This project provides a collection of Python scripts for real-time and static image facial processing using OpenCV and MediaPipe. It includes functionalities for face detection, pixelation (blurring), face swapping, and image upscaling.

---

## 📂 Project Structure

```bash
.
├── debug.py              # Test if a specific image file loads properly
├── faceDetection.py      # Real-time face detection via webcam
├── faceSwap.py           # Real-time and static face swapping using MediaPipe
├── faceBlur.py           # Face pixelation in live or static mode
├── imgUpscale.py         # Image upscaling script using Lanczos interpolation
└── README.md             # This documentation file
```

---

## ⚙️ Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

Install them with:

```bash
pip install opencv-python mediapipe numpy
```

---

## 📜 Script Descriptions

### `debug.py`

Checks if the image file `face_photo.jpg` is successfully loaded.

```bash
python debug.py
```

**Output:** `True` if the image failed to load, `False` otherwise.

---

### `faceDetection.py`

Performs real-time face detection using Haar Cascades and your webcam.

```bash
python faceDetection.py
```

**Features:**
- Webcam feed
- Real-time face bounding box drawing
- Press `q` to quit

---

### `faceSwap.py`

Performs face swapping between two detected faces or between live and static input.

```bash
# Live-to-live face swap
python faceSwap.py

# Static-to-live face swap
python faceSwap.py path/to/face.jpg
```

**Dependencies:**
- MediaPipe for 3D landmark detection
- NumPy and OpenCV for transformations and blending

**Notes:**
- Uses affine transform + seamlessClone
- Press `q` to quit

---

### `faceBlur.py`

Pixelates faces in real-time webcam feed or in static images.

```bash
python faceBlur.py
```

**Options:**
- Menu-driven interface: live mode or photo mode
- In live mode:  
  - Press `s` to save the frame  
  - Press `q` to quit
- In photo mode:  
  - Enter path to the image to pixelate  
  - Output saved as `pixelated_output.jpg`

---

### `imgUpscale.py`

Upscales an image using high-quality interpolation.

```bash
python imgUpscale.py
```

**Default Behavior:**
- Attempts to load `group2.jpg`
- Scales the image by a factor of 3
- Saves the result as `upscaled_output.jpg`

To customize:

```python
upscale_image("your_image.jpg", scale=2)
```

---

## 📌 To Do

- Add face tracking across frames
- Support for multiple faces per swap session
- Integrate GUI for all tools

---

## 🧑‍💻 Author

Andrej - Student developer and computer vision enthusiast

---
