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

image_queue = queue.Queue(maxsize=10)
captured_image_queue = queue.Queue(maxsize=1)
camera_lock = threading.Lock()
inference_queue = queue.Queue(maxsize=10)

def initialize_camera(port=CAM_PORT):
    camera = cv2.VideoCapture(port, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")
    return camera

def capture_image(camera):
    for _ in range(5):
        camera.grab()
    ret, frame = camera.read()
    if not ret:
        print("Warning: Failed to read frame")
        time.sleep(0.1)
        return None
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

def capture_thread(camera, stop_event):
    while not stop_event.is_set():
        try:
            start_time = time.time()
            with camera_lock:
                frame = capture_image(camera)
            if frame is not None:
                try:
                    inference_queue.put_nowait(frame)
                except queue.Full:
                    print("[Capture] Inference queue full, skipping frame")
            elapsed = time.time() - start_time
            print(f"[Capture] Time: {elapsed:.3f}s")
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Capture error: {e}")
            break

def inference_thread(model, stop_event):
    while not stop_event.is_set():
        try:
            start_time = time.time()
            frame = inference_queue.get(timeout=0.5)
            preprocessed_frame = preprocess(frame)
            output = predict(model, preprocessed_frame)
            predicted_label = LABELS[np.argmax(output)]
            print(f"[Inference] Predicted: {predicted_label}")
            try:
                image_queue.put_nowait((frame, predicted_label))
            except queue.Full:
                print("[Inference] Image queue full, skipping frame")
            elapsed = time.time() - start_time
            print(f"[Inference] Time: {elapsed:.3f}s")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Inference error: {e}")
            break

def display_images(stop_event, display_interval=0.1):
    captured_window_name = "Captured Object"
    detection_window_name = "Live Detection"
    while not stop_event.is_set():
        try:
            frame, predicted_label = image_queue.get(timeout=1)
            cv2.imshow(detection_window_name, frame)
            if not captured_image_queue.empty():
                captured_frame, captured_label = captured_image_queue.get()
                cv2.putText(
                    captured_frame, captured_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow(captured_window_name, captured_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop_event.set()
                break
            time.sleep(display_interval)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Display error: {e}")
            break

def load_tflite_model(model_path=MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main():
    camera = initialize_camera()
    model = load_tflite_model()
    dobot = DobotControl("dll/DobotDll.dll")
    dobot.connect("COM8")
    dobot.center()
    stop_event = threading.Event()
    capture = threading.Thread(target=capture_thread, args=(camera, stop_event))
    inference = threading.Thread(target=inference_thread, args=(model, stop_event))
    display = threading.Thread(target=display_images, args=(stop_event,))
    capture.daemon = True
    inference.daemon = True
    display.daemon = True
    capture.start()
    inference.start()
    display.start()
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
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Main loop error: {e}")
            break
    stop_event.set()
    capture.join(timeout=1)
    inference.join(timeout=1)
    display.join(timeout=1)
    with camera_lock:
        camera.release()
    cv2.destroyAllWindows()
    dobot.disconnect()

if __name__ == "__main__":
    main()

