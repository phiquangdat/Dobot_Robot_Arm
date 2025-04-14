import numpy as np
import cv2
import tensorflow as tf
import pydobot
import threading
import queue
from copy import deepcopy
import time

# [Your existing constants, Dobot setup, and other functions remain unchanged]
CAM_PORT = 0
MODEL_PATH = "ei-sort_defects-transfer-learning-tensorflow-lite-float32-model (5).lite"
LABELS = ("circle dirty", "circle ok", "nothing", "square dirty", "square ok", "triangle dirty", "triangle ok")

device = pydobot.Dobot(port='COM8')
x, y, z, r = 180, 0, -0, 0
start_z = 0
delta_z = 67
close_dist = 20
speed = 300
acceleration = 300
device.speed(speed, acceleration)

def vacuum_on():
    device.suck(True)

def vacuum_off():
    device.suck(False)

def wait(ms):
    device.wait(ms)

def center():
    device.move_to(180, 0, start_z + 20, r, wait=True)

def left45():
    device.move_to(150, 147, start_z, r, wait=True)

def right45():
    device.move_to(150, -147, start_z, r, wait=True)

def down():
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()
    device.move_to(x1, y1, start_z - delta_z - 5, r, wait=True)

def almost_down():
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()
    device.move_to(x1, y1, start_z - delta_z + close_dist, r, wait=True)

def up():
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()
    device.move_to(x1, y1, start_z, r, wait=True)

def fetch_object():
    vacuum_on()
    almost_down()
    down()
    up()

def release_object():
    almost_down()
    vacuum_off()
    up()

def initialize_camera(port=CAM_PORT):
    camera = cv2.VideoCapture(port, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")
    return camera

def load_tflite_model(model_path=MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def capture_image(camera):
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

def preprocess(frame, alpha=1, beta=50):
    processed = cv2.convertScaleAbs(frame)
    processed = cv2.resize(processed, (160, 160))
    processed = processed / 255.0
    processed = np.expand_dims(processed, axis=0).astype(np.float32)
    return processed

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Queues and lock
image_queue = queue.Queue(maxsize=10)
captured_image_queue = queue.Queue(maxsize=1)
camera_lock = threading.Lock()

def capture_and_process(camera, model, stop_event):
    while not stop_event.is_set():
        try:
            with camera_lock:
                frame = capture_image(camera)
            preprocessed_frame = preprocess(frame)
            output = predict(model, preprocessed_frame)
            predicted_label = LABELS[np.argmax(output)]
            print(f"[Capture] Predicted: {predicted_label}")
            try:
                image_queue.put_nowait((frame, predicted_label))
            except queue.Full:
                print("[Capture] Queue full, skipping frame")
                continue
            time.sleep(0.05)
        except Exception as e:
            print(f"Capture error: {e}")
            break

def display_images(stop_event):
    """Thread to handle image display with labels."""
    captured_window_name = "Captured Object"
    detection_window_name = "Live Detection"
    
    while not stop_event.is_set():
        try:
            # Get latest frame and label
            frame, predicted_label = image_queue.get(timeout=1)
            
            # Show live detection feed
            cv2.imshow(detection_window_name, frame)
            
            # Show captured image with label if available
            if not captured_image_queue.empty():
                captured_frame, captured_label = captured_image_queue.get()
                # Add label to the captured frame
                cv2.putText(
                    captured_frame,
                    captured_label,
                    (10, 30),  # Position (top-left corner)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # Font scale
                    (0, 255, 0),  # Green color
                    2,  # Thickness
                    cv2.LINE_AA
                )
                cv2.imshow(captured_window_name, captured_frame)
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                stop_event.set()
                break
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Display error: {e}")
            break

def move_bin1(frame, label):
    """Move object to bin 1 and show in new window with label."""
    print(f"MOVING: {label}")
    captured_image_queue.put((deepcopy(frame), label))  # Store frame and label
    fetch_object()
    right45()
    release_object()
    center()

def move_bin2(frame, label):
    """Move object to bin 2 and show in new window with label."""
    print(f"MOVING: {label}")
    captured_image_queue.put((deepcopy(frame), label))  # Store frame and label
    fetch_object()
    left45()
    release_object()
    center()

def main():
    camera = initialize_camera()
    model = load_tflite_model()
    
    center()
    
    # Threading setup
    stop_event = threading.Event()
    
    capture_thread = threading.Thread(target=capture_and_process, args=(camera, model, stop_event))
    capture_thread.daemon = True
    capture_thread.start()
    
    display_thread = threading.Thread(target=display_images, args=(stop_event,))
    display_thread.daemon = True
    display_thread.start()
    
    # Main loop
    while not stop_event.is_set():
        try:
            # Get latest prediction
            frame, predicted_label = image_queue.get(timeout=1)
            print(f"[Main] Processing: {predicted_label}")
            
            # Handle Dobot actions
            if predicted_label == 'circle ok':
                move_bin2(frame, predicted_label)
            elif predicted_label == 'circle dirty':
                move_bin1(frame, predicted_label)
            elif predicted_label == 'square dirty':
                move_bin1(frame, predicted_label)
            elif predicted_label == 'square ok':
                move_bin2(frame, predicted_label)
            elif predicted_label == 'triangle ok':
                move_bin2(frame, predicted_label)
            elif predicted_label == 'triangle dirty':
                move_bin1(frame, predicted_label)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Main loop error: {e}")
            break
    
    # Cleanup
    stop_event.set()
    capture_thread.join(timeout=1)
    display_thread.join(timeout=1)
    with camera_lock:
        camera.release()
    cv2.destroyAllWindows()
    device.close()

if __name__ == "__main__":
    main()