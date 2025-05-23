import ctypes
import os
from ctypes import c_int, c_float, c_bool, c_char_p, c_uint, byref, Structure
import time

class DobotPose(Structure):
    """Structure to hold Dobot's pose (x, y, z, r)."""
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("r", c_float)]

class DobotControl:
    """
    Python wrapper for DobotDll.dll to control the Dobot Magician robotic arm.
    
    Dependencies:
        - msvcp120.dll, msvcr120.dll (Microsoft Visual C++ 2013 Redistributable)
        - Qt5Core.dll, Qt5Network.dll, Qt5SerialPort.dll (Qt framework)
        These must be in the same directory as DobotDll.dll or in the system PATH.
    """
    
    def __init__(self, dll_path="DobotDll.dll"):
        """
        Initialize the Dobot controller by loading DobotDll.dll.
        
        Args:
            dll_path (str): Path to DobotDll.dll. Defaults to 'DobotDll.dll'.
        
        Raises:
            FileNotFoundError: If DobotDll.dll or dependencies are missing.
            WindowsError: If the DLL fails to load.
        """
<<<<<<< HEAD
        # Check for DLL and dependencies
=======
>>>>>>> 5f302d8 (Create dll files)
        dll_dir = os.path.dirname(dll_path) or os.getcwd()
        required_dlls = ["msvcp120.dll", "msvcr120.dll", "Qt5Core.dll", "Qt5Network.dll", "Qt5SerialPort.dll"]
        for dll in required_dlls:
            if not os.path.exists(os.path.join(dll_dir, dll)) and not os.path.exists(os.path.join("C:\\Windows\\System32", dll)):
                raise FileNotFoundError(f"Dependency {dll} not found in {dll_dir} or system PATH")
        
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DobotDll.dll not found at {dll_path}")
        
        try:
            self.dll = ctypes.WinDLL(dll_path)
        except WindowsError as e:
            raise WindowsError(f"Failed to load DobotDll.dll: {e}. Ensure dependencies are present.")
        
<<<<<<< HEAD
        # Define DobotDll.dll function signatures
=======
>>>>>>> 5f302d8 (Create dll files)
        self.dll.ConnectDobot.argtypes = [c_char_p, c_uint, ctypes.POINTER(c_int)]
        self.dll.ConnectDobot.restype = c_int
        self.dll.DisconnectDobot.argtypes = []
        self.dll.DisconnectDobot.restype = None
        self.dll.SetPTPCmd.argtypes = [c_int, c_int, ctypes.POINTER(DobotPose), c_bool]
        self.dll.SetPTPCmd.restype = c_int
        self.dll.SetEndEffectorSuctionCup.argtypes = [c_int, c_bool, c_bool]
        self.dll.SetEndEffectorSuctionCup.restype = c_int
        self.dll.SetHOMECmd.argtypes = [c_int]
        self.dll.SetHOMECmd.restype = c_int
        self.dll.GetPose.argtypes = [c_int, ctypes.POINTER(DobotPose)]
        self.dll.GetPose.restype = c_int
        self.dll.SetPTPCommonParams.argtypes = [c_int, c_float, c_float, c_bool]
        self.dll.SetPTPCommonParams.restype = c_int
        
        self.connected = False
        self.api = c_int(0)
<<<<<<< HEAD

    def connect(self, port=None, baudrate=115200):
        """
        Connect to the Dobot Magician.
        
        Args:
            port (str): Serial port (e.g., 'COM8'). If None, auto-detects.
            baudrate (int): Baud rate. Defaults to 115200.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        # Auto-detect port if not provided
=======
        self.start_z = 0
        self.delta_z = 67
        self.close_dist = 20
        self.r = 0

    def connect(self, port=None, baudrate=115200):
>>>>>>> 5f302d8 (Create dll files)
        if port is None:
            try:
                import serial.tools.list_ports
                ports = serial.tools.list_ports.comports()
                for p in ports:
                    try:
                        result = self.dll.ConnectDobot(p.device.encode(), baudrate, byref(self.api))
                        if result == 0:
                            port = p.device
                            break
                    except:
                        continue
                else:
                    return False
            except ImportError:
                return False
        else:
            result = self.dll.ConnectDobot(port.encode(), baudrate, byref(self.api))
        
        if result == 0:
            self.connected = True
<<<<<<< HEAD
            # Set default speed and acceleration
=======
>>>>>>> 5f302d8 (Create dll files)
            self.dll.SetPTPCommonParams(self.api, 300.0, 300.0, True)
        return self.connected

    def disconnect(self):
<<<<<<< HEAD
        """Disconnect from the Dobot Magician."""
        if self.connected:
            self.dll.DisconnectDobot()
            self.connected = False

    def move_to(self, x, y, z, r, mode=0):
        """
        Move the Dobot to the specified coordinates.
        
        Args:
            x (float): X-coordinate in mm.
            y (float): Y-coordinate in mm.
            z (float): Z-coordinate in mm.
            r (float): Rotation in degrees.
            mode (int): PTP mode (0 = JUMP_XYZ, 1 = MOVJ_XYZ, 2 = MOVL_XYZ).
        
        Returns:
            bool: True if successful, False otherwise.
        """
=======
        if self.connected:
            self.dll.DisconnectDobot()
            self.connected = False
            time.sleep(0.5)  # Ensure clean disconnect

    def move_to(self, x, y, z, r, mode=0):
>>>>>>> 5f302d8 (Create dll files)
        if not self.connected:
            return False
        pose = DobotPose(x, y, z, r)
        result = self.dll.SetPTPCmd(self.api, mode, byref(pose), True)
<<<<<<< HEAD
        return result == 0

    def vacuum_on(self):
        """
        Turn on the vacuum suction cup.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.connected:
            return False
        result = self.dll.SetEndEffectorSuctionCup(self.api, True, True)
        return result == 0

    def vacuum_off(self):
        """
        Turn off the vacuum suction cup.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.connected:
            return False
        result = self.dll.SetEndEffectorSuctionCup(self.api, True, False)
        return result == 0

    def home(self):
        """
        Move the Dobot to its home position.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.connected:
            return False
        result = self.dll.SetHOMECmd(self.api)
        return result == 0

    def get_pose(self):
        """
        Get the current pose of the Dobot.
        
        Returns:
            tuple: (x, y, z, r) if successful, None otherwise.
        """
=======
        time.sleep(0.1)  # Stabilize USB
        return result == 0

    def vacuum_on(self):
        if not self.connected:
            return False
        result = self.dll.SetEndEffectorSuctionCup(self.api, True, True)
        time.sleep(0.1)
        return result == 0

    def vacuum_off(self):
        if not self.connected:
            return False
        result = self.dll.SetEndEffectorSuctionCup(self.api, True, False)
        time.sleep(0.1)
        return result == 0

    def home(self):
        if not self.connected:
            return False
        result = self.dll.SetHOMECmd(self.api)
        time.sleep(0.1)
        return result == 0

    def get_pose(self):
>>>>>>> 5f302d8 (Create dll files)
        if not self.connected:
            return None
        pose = DobotPose()
        result = self.dll.GetPose(self.api, byref(pose))
        if result == 0:
            return (pose.x, pose.y, pose.z, pose.r)
        return None

<<<<<<< HEAD
    def __del__(self):
        """Ensure disconnection on object destruction."""
        self.disconnect()
=======
    def center(self):
        self.move_to(180, 0, self.start_z + 20, self.r, mode=1)
        time.sleep(0.1)

    def left45(self):
        self.move_to(150, 147, self.start_z, self.r, mode=1)
        time.sleep(0.1)

    def right45(self):
        self.move_to(150, -147, self.start_z, self.r, mode=1)
        time.sleep(0.1)

    def down(self):
        pose = self.get_pose()
        if pose:
            x, y, _, r = pose
            self.move_to(x, y, self.start_z - self.delta_z - 5, r, mode=1)
        time.sleep(0.1)

    def almost_down(self):
        pose = self.get_pose()
        if pose:
            x, y, _, r = pose
            self.move_to(x, y, self.start_z - self.delta_z + self.close_dist, r, mode=1)
        time.sleep(0.1)

    def up(self):
        pose = self.get_pose()
        if pose:
            x, y, _, r = pose
            self.move_to(x, y, self.start_z, r, mode=1)
        time.sleep(0.1)

    def fetch_object(self):
        self.vacuum_on()
        self.almost_down()
        self.down()
        self.up()

    def release_object(self):
        self.almost_down()
        self.vacuum_off()
        self.up()

    def __del__(self):
        self.disconnect()

>>>>>>> 5f302d8 (Create dll files)
