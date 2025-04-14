import cv2

# Try different indices to find the external camera.
for index in range(10):  # Increase range if you have more connected devices
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        cap.release()
    else:
        print(f"No camera found at index {index}")
