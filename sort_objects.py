import numpy as np
import cv2
import tensorflow as tf
import threading
import queue
from copy import deepcopy
import time
from dobot_control import DobotControl

CAM_PORT = 0
MODEL_PATH = "ei-sort_defects-transfer-learning-tensorflow-lite-float32-model (5).lite"
LABELS = ("circle dirty", "circle ok", "nothing", "square dirty", "square ok", "triangle dirty", "triangle ok")

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
            time.sleep(0.05)
        except Exception as e:
            print(f"Capture error: {e}")
            break

def display_images(stop_event):
    captured_window_name = "Captured Object"
    detection_window_name = "Live Detection"
    
    while not stop_event.is_set():
        try:
            frame, predicted_label = image_queue.get(timeout=1)
            cv2.imshow(detection_window_name, frame)
            if not captured_image_queue.empty():
                captured_frame, captured_label = captured_image_queue.get()
                cv2.putText(
                    captured_frame,
                    captured_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(captured_window_name, captured_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop_event.set()
                break
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Display error: {e}")
            break

def main():
    camera = initialize_camera()
    model = load_tflite_model()
    dobot = DobotControl("dll/DobotDll.dll", port="COM8")
    
    dobot.center()
    
    stop_event = threading.Event()
    capture_thread = threading.Thread(target=capture_and_process, args=(camera, model, stop_event))
    capture_thread.daemon = True
    capture_thread.start()
    
    display_thread = threading.Thread(target=display_images, args=(stop_event,))
    display_thread.daemon = True
    display_thread.start()
    
    while not stop_event.is_set():
        try:
            frame, predicted_label = image_queue.get(timeout=1)
            print(f"[Main] Processing: {predicted_label}")
            
            if predicted_label == 'circle ok':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.left45()
                dobot.release_object()
                dobot.center()
            elif predicted_label == 'circle dirty':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.right45()
                dobot.release_object()
                dobot.center()
            elif predicted_label == 'square dirty':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.right45()
                dobot.release_object()
                dobot.center()
            elif predicted_label == 'square ok':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.left45()
                dobot.release_object()
                dobot.center()
            elif predicted_label == 'triangle ok':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.left45()
                dobot.release_object()
                dobot.center()
            elif predicted_label == 'triangle dirty':
                captured_image_queue.put((deepcopy(frame), predicted_label))
                dobot.fetch_object()
                dobot.right45()
                dobot.release_object()
                dobot.center()
            
        except queue Hannah exceptions:
            continue
        except Exception as e:
            print(f"Main loop error: {e}")
            break
    
    stop_event.set()
    capture_thread.join(timeout=1)
    display_thread.join(timeout=1)
    with camera_lock:
        camera.release()
    cv2.destroyAllWindows()
    dobot.disconnect()

if __name__ == "__main__":
    main()