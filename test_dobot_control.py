import unittest
import os
from dobot_control import DobotControl

class TestDobotControl(unittest.TestCase):
    """Unit tests for DobotControl using DobotDll.dll."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize the Dobot controller before running tests."""
        dll_path = "DobotDll.dll"
        if not os.path.exists(dll_path):
            raise unittest.SkipTest(f"DobotDll.dll not found at {dll_path}")
        cls.dobot = DobotControl(dll_path)

    def test_connect_disconnect(self):
        """Test connecting and disconnecting from the Dobot."""
        self.assertTrue(self.dobot.connect(), "Failed to connect")
        self.assertTrue(self.dobot.connected, "Connected flag not set")
        self.dobot.disconnect()
        self.assertFalse(self.dobot.connected, "Disconnected flag not cleared")

    def test_move_to(self):
        """Test moving to a specified position."""
        self.assertTrue(self.dobot.connect(), "Failed to connect")
        self.assertTrue(self.dobot.move_to(180, 0, 0, 0), "Failed to move")
        self.dobot.disconnect()

    def test_vacuum(self):
        """Test turning the vacuum suction cup on and off."""
        self.assertTrue(self.dobot.connect(), "Failed to connect")
        self.assertTrue(self.dobot.vacuum_on(), "Failed to turn on vacuum")
        self.assertTrue(self.dobot.vacuum_off(), "Failed to turn off vacuum")
        self.dobot.disconnect()

    def test_home(self):
        """Test moving to the home position."""
        self.assertTrue(self.dobot.connect(), "Failed to connect")
        self.assertTrue(self.dobot.home(), "Failed to move to home")
        self.dobot.disconnect()

    def test_get_pose(self):
        """Test retrieving the current pose."""
        self.assertTrue(self.dobot.connect(), "Failed to connect")
        pose = self.dobot.get_pose()
        self.assertIsNotNone(pose, "Failed to get pose")
        self.assertEqual(len(pose), 4, "Pose should have 4 components")
        self.dobot.disconnect()

    @classmethod
    def tearDownClass(cls):
        """Ensure disconnection after all tests."""
        cls.dobot.disconnect()

if __name__ == "__main__":
    unittest.main()
