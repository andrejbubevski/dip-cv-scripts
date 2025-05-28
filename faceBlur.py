import cv2
import numpy as np
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def pixelate_face(img, pixel_size=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        temp = cv2.resize(face_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = pixelated

    return img

def live_mode():
    cap = cv2.VideoCapture(0)
    print("Live Mode - Press 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror off = remove this line
        pixelated = pixelate_face(frame.copy(), pixel_size=45)
        cv2.imshow('Live Face Pixelation', pixelated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"pixelated_{timestamp}.jpg", pixelated)
            print(f"Saved pixelated_{timestamp}.jpg")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def photo_mode(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found.")
        return
    pixelated = pixelate_face(img, pixel_size=15)
    cv2.imshow("Pixelated Photo", pixelated)
    cv2.imwrite("pixelated_output.jpg", pixelated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Saved as pixelated_output.jpg")

def main_menu():
    print("\nPixelation Demo App")
    print("1. Live camera face pixelation")
    print("2. Pixelate faces in a photo")
    print("q. Quit")
    choice = input("Your choice: ").strip()

    if choice == '1':
        live_mode()
    elif choice == '2':
        path = input("Enter image path: ")
        photo_mode(path)
    elif choice == 'q':
        print("Exiting.")
    else:
        print("Invalid choice.")
        main_menu()

if __name__ == "__main__":
    main_menu()