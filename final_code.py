import numpy as np  # Import NumPy library for numerical operations
import cv2  # Import OpenCV library for image processing
import tensorflow as tf  # Import TensorFlow library for machine learning models
import pydobot, serial  # Import Pydobot and serial libraries for Dobot control
import threading
import queue
import time
# Constants
CAM_PORT = 0  # Define camera port (usually 0)
MODEL_PATH = "ei-sort_defects-transfer-learning-tensorflow-lite-float32-model (5).lite"  # Path to the TensorFlow Lite model file
LABELS = ("circle dirty", "circle ok", "nothing", "square dirty", "square ok", "triangle dirty", "triangle ok")  # Define classification labels

device = pydobot.Dobot(port='COM8')  # Connect to Dobot Magician, replace 'COM18' with the appropriate port
x, y, z, r = 180, 0, -0, 0  # Initial coordinates and rotation for Dobot
start_z = 0  # Initial Z-coordinate
delta_z = 67  # Z-coordinate change for calibration
close_dist = 20  # Distance for "almost down" position
speed = 300  # Movement speed
acceleration = 300  # Movement acceleration
device.speed(speed, acceleration)  # Set Dobot's speed and acceleration

class CameraThread:
    def __init__(self, port=CAM_PORT):
        self.camera = cv2.VideoCapture(port, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame)
            time.sleep(0.02)  # Limit to ~50 FPS to reduce USB contention

    def get_frame(self):
        # Flush buffer
        for _ in range(5):
            self.camera.read()
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.camera.release()
        
def vacuum_on():  # Turn on the vacuum pump
    device.suck(True)  # Activate the vacuum pump

def vacuum_off():  # Turn off the vacuum pump
    device.suck(False)  # Deactivate the vacuum pump

def wait(ms):  # Implement a wait function
    device.wait(ms)  # Dobot waits for the specified time in milliseconds

def center():  # Move Dobot to the center position
    device.move_to(180, 0, start_z + 20, r, wait=True)  # Move Dobot to center and slightly up to avoid being in the camera frame

def left45():  # Rotate Dobot 45 degrees to the left
    device.move_to(150, 147, start_z, r, wait=True)  # Move Dobot 45 degrees left

def right45():  # Rotate Dobot 45 degrees to the right
    device.move_to(150, -147, start_z, r, wait=True)  # Move Dobot 45 degrees right

def down():  # Move Dobot to the down position
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()  # Get current position
    device.move_to(x1, y1, start_z - delta_z -5, r, wait=True)  # Move Dobot down

def almost_down():  # Move Dobot to almost down position
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()  # Get current position
    device.move_to(x1, y1, start_z - delta_z + close_dist, r, wait=True)  # Move Dobot almost down

def up():  # Move Dobot to the up position
    (x1, y1, z1, r, j1, j2, j3, j4) = device.pose()  # Get current position
    device.move_to(x1, y1, start_z, r, wait=True)  # Move Dobot up

def fetch_object():  # Pick up an object
    vacuum_on()  # Turn on vacuum pump
    almost_down()  # Move near the surface
    down()  # Move fully down
    up()  # Move back up

def release_object():  # Release an object
    almost_down()  # Move near the surface
    vacuum_off()  # Turn off vacuum pump
    up()  # Move back up

# From here, you can integrate image recognition results with Dobot control

# Camera and model initialization
def initialize_camera(port=CAM_PORT):  # Initialize the camera
    return cv2.VideoCapture(port, cv2.CAP_DSHOW)  # Return the camera capture object

def load_tflite_model(model_path=MODEL_PATH):  # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)  # Load the TFLite model
    interpreter.allocate_tensors()  # Allocate tensors for the interpreter
    return interpreter  # Return the model interpreter

def capture_image(camera):  # Capture an image from the camera
    ret, frame = camera.read()  # Read an image from the camera
    if not ret:  # If image capture fails
        raise RuntimeError("Failed to capture image")  # Raise a RuntimeError
    return frame  # Return the captured image

def preprocess(frame, alpha=1, beta=50):  # Preprocess the image for prediction
    processed = cv2.convertScaleAbs(frame)  # Apply scale conversion to the image
    processed = cv2.resize(processed, (160, 160))  # Resize image to the model's expected size
    processed = processed / 255.0  # Normalize pixel values to range [0, 1]
    processed = np.expand_dims(processed, axis=0).astype(np.float32)  # Add batch dimension and ensure float32 type
    return processed  # Return the preprocessed image

def predict(interpreter, image):  # Make a prediction using the TensorFlow Lite model
    input_details = interpreter.get_input_details()  # Get model input details
    output_details = interpreter.get_output_details()  # Get model output details
    interpreter.set_tensor(input_details[0]['index'], image)  # Set input tensor
    interpreter.invoke()  # Run the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])  # Get output data
    return output_data  # Return prediction results

def return_prediction(camera, model):  # Get prediction and captured frame
    frame = capture_image(camera)  # Capture an image from the camera
    preprocessed_frame = preprocess(frame)  # Preprocess the captured image
    cv2.imshow("Preprocessed Frame", frame)  # Display the preprocessed image
    output = predict(model, preprocessed_frame)  # Make a prediction on the preprocessed image
    predicted_label = LABELS[np.argmax(output)]  # Get the predicted label
    return predicted_label, frame  # Return the label and frame

def show_captured_frame(captured_frame):
    if captured_frame is not None:
        cv2.imshow("Preprocessed Frame", captured_frame)
        cv2.waitKey(50)

def close_camera(camera):  # Close the camera
    camera.release()  # Release the camera resource

def move_bin1(camera_thread, frame, label):
    show_captured_frame(frame)
    print("MOVING:", label)
    fetch_object()
    right45()
    release_object()
    center()
    return camera_thread

def move_bin2(camera_thread, frame, label):
    show_captured_frame(frame)
    print("MOVING:", label)
    fetch_object()
    left45()
    release_object()
    center()
    return camera_thread

# Main program
def main():  # Main function
    camera_thread = CameraThread()  # Initialize the camera
    model = load_tflite_model()  # Load the TensorFlow Lite model
    
    center()  # Move Dobot to the center position

    try:
        camera_thread.start()
        center()
        while True:  # Start an infinite loop
            predicted_label, captured_frame = return_prediction(camera, model)  # Get prediction and frame
            print("Predicted label:", predicted_label)  # Print the predicted label

            # Define how Dobot reacts based on the predicted label
            if predicted_label == 'circle ok':
                camera = move_bin2(camera, captured_frame, predicted_label)  # Move to center (no action yet)
                
            elif predicted_label == 'circle dirty':
                camera = move_bin1(camera, captured_frame, predicted_label)    # Move to center (no action yet)
                
            elif predicted_label == 'square dirty':
                camera = move_bin1(camera, captured_frame, predicted_label)  # Move to bin 1
                
            elif predicted_label == 'square ok':
                camera = move_bin2(camera, captured_frame, predicted_label)  # Move to bin 2
                
            elif predicted_label == 'triangle ok':
                camera = move_bin2(camera, captured_frame, predicted_label)  # Move to center (no action yet)
                
            elif predicted_label == 'triangle dirty':
                camera = move_bin1(camera, captured_frame, predicted_label)    # Move to center (no action yet)
        
            # Add more conditions and actions as needed
            if cv2.waitKey(1) & 0xFF == 27:  # If ESC key is pressed
                center()  # Move to center
                break  # Exit the loop

    # Release resources at the end
    finally:
        camera_thread.stop()  # Release the camera
        cv2.destroyAllWindows()  # Close all OpenCV windows
        device.close()  # Close the Dobot connection

if __name__ == "__main__":  # If this script is run directly
    main()  # Execute the main function
