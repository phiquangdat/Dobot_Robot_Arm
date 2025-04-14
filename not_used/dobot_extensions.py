# Import necessary modules
import pydobot   # The pydobot module is used to interact with Dobot robotic arms.
import math      # Math module is used for mathematical operations such as calculations involving π.
import struct    # Struct module is used to work with binary data.

from pydobot.message import Message  # Importing the Message class to handle commands sent to the Dobot.

# Define constants for general purpose ports.
PORT_GP1 = 0x00  # GP1 port identifier.
PORT_GP2 = 0x01  # GP2 port identifier.
PORT_GP4 = 0x02  # GP4 port identifier.
PORT_GP5 = 0x03  # GP5 port identifier.

# Define a custom Dobot class that inherits from the pydobot.Dobot class.
class Dobot(pydobot.Dobot):
    # Method to control the conveyor belt's distance and speed.
    def conveyor_belt_distance(self, speed_mm_per_sec, distance_mm, direction=1, interface=0):
        # Ensure the speed does not exceed 100 mm/s for safety or hardware limitations.
        if speed_mm_per_sec > 100:
            raise pydobot.dobot.DobotException("Speed must be <= 100 mm/s")

        # Constants for the conveyor belt's mechanics.
        MM_PER_REV = 34 * math.pi  # Determines the millimeters per revolution (based on measured values, 34 mm works well).
        STEP_ANGLE_DEG = 1.8  # Step angle of the stepper motor in degrees.
        STEPS_PER_REV = 360.0 / STEP_ANGLE_DEG * 10.0 * 16.0 / 2.0  
        # Calculates the total steps per revolution using the motor's specifications and a conversion factor.

        # Calculate the distance and speed in terms of motor steps.
        distance_steps = distance_mm / MM_PER_REV * STEPS_PER_REV
        speed_steps_per_sec = speed_mm_per_sec / MM_PER_REV * STEPS_PER_REV * direction

        # Send the calculated parameters to the stepper motor and return the command index for tracking.
        return self._extract_cmd_index(self._set_stepper_motor_distance(
            int(speed_steps_per_sec), int(distance_steps), interface))

    # Method to enable or disable a color sensor connected to a specific port.
    def set_color(self, enable=True, port=PORT_GP2, version=0x1):
        msg = Message()  # Create a new command message.
        msg.id = 137     # Set the ID for the color sensor command.
        msg.ctrl = 0x03  # Control flag indicating a request to set the sensor's state.
        msg.params = bytearray([])  # Initialize parameters as an empty byte array.
        msg.params.extend(bytearray([int(enable)]))  # Add enable/disable flag to the message.
        msg.params.extend(bytearray([port]))         # Specify the port where the sensor is connected.
        msg.params.extend(bytearray([version]))      # Specify the version of the sensor.
        return self._extract_cmd_index(self._send_command(msg))  # Send the command and return its index.

    # Method to retrieve color data from the sensor.
    def get_color(self, port=PORT_GP2, version=0x1):
        msg = Message()  # Create a new command message.
        msg.id = 137     # Set the ID for the color sensor command.
        msg.ctrl = 0x00  # Control flag indicating a request to read sensor data.
        msg.params = bytearray([])  # Initialize parameters as an empty byte array.
        msg.params.extend(bytearray([port]))         # Specify the port where the sensor is connected.
        msg.params.extend(bytearray([0x01]))         # Add a parameter (purpose unclear in this context).
        msg.params.extend(bytearray([version]))      # Specify the version of the sensor.
        
        # Send the command and retrieve the response from the Dobot.
        response = self._send_command(msg)
        print(response)  # Print the raw response for debugging purposes.

        # Unpack the binary response to extract RGB color values.
        r = struct.unpack_from('?', response.params, 0)[0]  # Extract the red color value.
        g = struct.unpack_from('?', response.params, 1)[0]  # Extract the green color value.
        b = struct.unpack_from('?', response.params, 2)[0]  # Extract the blue color value.
        return [r, g, b]  # Return the RGB values as a list.
