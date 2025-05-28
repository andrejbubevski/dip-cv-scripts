import cv2
import numpy as np
import mediapipe as mp
import sys

mp_face_mesh = mp.solutions.face_mesh
# Use static_image_mode=False for live frames, but we will re-initialize for static image processing
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)

def get_face_landmarks(image, static=False):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # If static image, reinitialize face_mesh for better detection
    if static:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True) as static_face_mesh:
            results = static_face_mesh.process(rgb)
    else:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    return [np.array([[p.x * image.shape[1], p.y * image.shape[0]] for p in face_landmarks.landmark])
            for face_landmarks in results.multi_face_landmarks]

def draw_convex_mask(image, points):
    hull = cv2.convexHull(points.astype(np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask, hull

def apply_swap(image, landmarks1, landmarks2):
    indices = [33, 263, 1]  # left eye, right eye, nose tip
    pts1 = landmarks1[indices].astype(np.float32)
    pts2 = landmarks2[indices].astype(np.float32)

    M = cv2.getAffineTransform(pts2, pts1)
    warped = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    mask, hull = draw_convex_mask(image, landmarks1)
    center = tuple(np.mean(hull[:, 0], axis=0).astype(int))
    h, w = image.shape[:2]
    center = (min(max(center[0], 0), w - 1), min(max(center[1], 0), h - 1))
    try:
        result = cv2.seamlessClone(warped, image, mask, center, cv2.NORMAL_CLONE)
    except cv2.error as e:
        print("[WARN] seamlessClone failed:", e)
        result = image
    return result

# To run static image swap: python faceSwap.py path/to/face.jpg
cap = cv2.VideoCapture(0)

use_static_image = len(sys.argv) > 1  # pass image path as command-line argument

if use_static_image:
    static_image = cv2.imread(sys.argv[1])
    if static_image is None:
        print("Error: Could not read static image.")
        sys.exit(1)

    # Resize for better face detection
    static_image = cv2.resize(static_image, (640, 480))

    static_landmarks = get_face_landmarks(static_image, static=True)
    print("Static landmarks:", static_landmarks)
    if not static_landmarks or len(static_landmarks) < 1:
        print("Error: Could not detect face in static image.")
        sys.exit(1)
    static_landmarks = static_landmarks[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)  # Flip horizontally to remove mirror effect

    landmarks = get_face_landmarks(frame, static=False)
    if landmarks and len(landmarks) >= 1:
        try:
            if use_static_image:
                frame = apply_swap(frame, landmarks[0], static_landmarks)
            elif len(landmarks) >= 2:
                frame = apply_swap(frame, landmarks[0], landmarks[1])
        except Exception as e:
            print("[ERROR] Face swap failed:", e)

    cv2.imshow("Live Face Swap (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()