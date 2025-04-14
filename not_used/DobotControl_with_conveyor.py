# Import necessary modules for threading, Dobot DLL interactions, and defining robotic functionalities.
import threading          # Provides support for threading to perform tasks concurrently.
import DobotDllType as dType  # A Dobot-specific library for interacting with Dobot robots.

# Initialize variables to store belt speed and the presence of objects detected by sensors.
belt_speed = None  
object_found = None

# Print a debug message indicating the start of the program.
print('0')

# Define a dictionary to map Dobot connection statuses to human-readable strings.
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

# Print another debug message indicating progress in the program.
print(1)

# Load the Dobot DLL and get the CDLL object for performing operations.
api = dType.load()

# Connect to the Dobot robot. The connection function returns a status code.
state = dType.ConnectDobot(api, "", 115200)[0]  # Connect to Dobot using default port settings.
print("Connect status:", CON_STR[state])  # Display the connection status in a human-readable format.
print(2)

# If the connection was successful (NoError status), execute the following commands.
if (state == dType.DobotConnect.DobotConnect_NoError):
    print(3)
    
    # Clear the command queue to ensure no residual commands are executed.
    dType.SetQueuedCmdClear(api)
    
    # Set motion parameters asynchronously for the Dobot.
    dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued=1)  # Set HOME positions.
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)  # Set joint speed parameters.
    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)  # Set common speed and acceleration parameters.

    # Command the robot to perform its "Home" procedure asynchronously.
    dType.SetHOMECmd(api, temp=0, isQueued=1)

    # Perform Point-To-Point (PTP) motion asynchronously in a loop.
    for i in range(0, 5):
        # Alternate between positive and negative offsets based on loop index.
        if i % 2 == 0:
            offset = 10
        else:
            offset = -10
        # Send PTP motion commands to the Dobot and get the last command index.
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200 + offset, offset, offset, offset, isQueued=1)[0]

    # Retrieve the current pose of the Dobot robot and print it for debugging purposes.
    current_pose = dType.GetPose(api)
    print(current_pose)

    # Command Dobot to move to specific coordinates using precise PTP motion.
    dType.SetPTPCmdEx(api, 2, 275.02, 24.97, 25, current_pose[3], 1)

    # Set the speed of the conveyor belt in mm/s.
    belt_speed = 20
    # Enable the infrared sensor connected to the Dobot.
    dType.SetInfraredSensor(api, 1, 1, 1)
    
    # Calculate step and velocity parameters for the conveyor belt motor based on its specifications.
    STEP_PER_CRICLE = 360.0 / 1.8 * 10.0 * 16.0  # Steps per revolution based on motor step angle and mechanical ratio.
    MM_PER_CRICLE = 3.1415926535898 * 36.0  # Millimeters per revolution based on wheel circumference.
    vel = float(belt_speed) * STEP_PER_CRICLE / MM_PER_CRICLE  # Calculate speed in steps per second.
    dType.SetEMotorEx(api, 0, 1, int(vel), 1)  # Command the conveyor belt motor to move.

    # Start executing queued commands.
    dType.SetQueuedCmdStartExec(api)

    # Print a debug message indicating the beginning of the loop.
    print("Before loop")

    # Enter an infinite loop to detect objects and handle their presence.
    while True:
        print(4)
        
        # Check if the infrared sensor detects an object.
        object_found = dType.GetInfraredSensor(api, 1)[0]
        
        if object_found:
            # If an object is found, stop the conveyor belt.
            STEP_PER_CRICLE = 360.0 / 1.8 * 10.0 * 16.0
            MM_PER_CRICLE = 3.1415926535898 * 36.0
            vel = float(0) * STEP_PER_CRICLE / MM_PER_CRICLE
            dType.SetEMotorEx(api, 0, 0, int(vel), 1)
            
            # Activate the suction cup to pick up the object.
            dType.SetEndEffectorSuctionCupEx(api, 1, 1)
            
            # Move the Dobot to various predefined positions while handling the object.
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 2, 277.98, 25, 16, current_pose[3], 1)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 2, 277.98, 25, 30, current_pose[3], 1)
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 2, 278.46, 99.24, 20, current_pose[3], 1)
            
            # Deactivate the suction cup to release the object.
            dType.SetEndEffectorSuctionCupEx(api, 0, 1)
        else:
            # If no object is found, continue with conveyor belt motion.
            current_pose = dType.GetPose(api)
            dType.SetPTPCmdEx(api, 2, 275.02, 24.97, 25, current_pose[3], 1)
            STEP_PER_CRICLE = 360.0 / 1.8 * 10.0 * 16.0
            MM_PER_CRICLE = 3.1415926535898 * 36.0
            vel = float(belt_speed) * STEP_PER_CRICLE / MM_PER_CRICLE
            dType.SetEMotorEx(api, 0, 1, int(vel), 1)
        
        # Execute the queued commands.
        dType.SetQueuedCmdStartExec(api)

    # Stop the execution of queued commands when necessary.
    dType.SetQueuedCmdStopExec(api)

# Disconnect from the Dobot robot when finished.
dType.DisconnectDobot(api)
