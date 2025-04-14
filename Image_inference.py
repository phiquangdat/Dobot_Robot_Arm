import numpy as np
import time
from nltk import flatten
import cv2
import tensorflow as tf

labels = ("circle dirty", "circle ok", "nothing", "square dirty", "square ok", "triangle dirty", "triangle ok")


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter("ei-sort_defects-transfer-learning-tensorflow-lite-float32-model (5).lite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']

print("***** INPUT DETAILS *****")
print(input_details)

print("***** OUTPUT DETAILS *****")
print(output_details)

cam_port = 0
#cam = cv2.VideoCapture(cam_port) #, cv2.CAP_DSHOW) 



while True:
    cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW) 
    
    # reading the input using the camera 
    result, image = cam.read() 

    if not result:
        print("Failed to grab frame")
    else:

        # Method 2: Using convertScaleAbs (handles underflow/overflow better)
        alpha = 11  # Contrast control (1.0-3.0)
        beta = 150  # Brightness control (0-100)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        cv2.imshow("test", image)  # Display the frame
        #time.sleep(2)
        cv2.waitKey(1)
        
        #cv2.destroyWindow("test") 
        # img_name = f"opencv_frame.png"
        # cv2.imwrite(img_name, image)  # Save the image
        # print(f"{img_name} written!")



    # Resize the image to the expected size for MobileNetV2
    image = cv2.resize(image, (160, 160))  # Change this if different for your model

    # Convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("test", image)  # Display the frame
    # cv2.waitKey(0) 
    # cv2.destroyWindow("test") 

    # Scale pixel values from [0, 255] to [-1, 1]
    image = (image.astype(np.float32))  / 255
    #print(image)

    # Add a batch dimension
    image = np.expand_dims(image, axis=0)


    #image = np.expand_dims(image, axis=-1)  # Now shape will be (1, 96, 96, 1) if your model expects this  ********* FOR GRAYSCALE IMAGE !!!!!!!!!!!!!!!!!!!!!!

    #image = flatten(image)
    # print(image)

    # Set the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the inference
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"{output_data[0][0]:.3f}")
    print(f"{output_data[0][1]:.3f}")
    print(f"{output_data[0][2]:.3f}")
    print(f"{output_data[0][3]:.3f}")
    print(f"{output_data[0][4]:.3f}")
    print(f"{output_data[0][5]:.3f}")
    print(f"{output_data[0][6]:.3f}")

    # print(f"{output_data[0]:.3f}")

    #print(max(output_data[0]))
    index_of_max = np.argmax(output_data)  # Get the index of the maximum value

    # Print the corresponding label
    print("Predicted label:", labels[index_of_max])

    # Release the camera and close the window
    cam.release()
    #cv2.destroyAllWindows()