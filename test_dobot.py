import unittest
import os
from dobot_control import DobotControl

class TestDobotControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dll_path = os.path.join(os.path.dirname(__file__), "DobotDll.dll")
        cls.dobot = DobotControl(dll_path, port="COM8")
    
    @classmethod
    def tearDownClass(cls):
        cls.dobot.disconnect()
    
    def test_connect(self):
        self.assertTrue(True)  # Connection tested in setUpClass
    
    def test_move_to_center(self):
        try:
            self.dobot.center()
            self.assertTrue(True)
        except RuntimeError:
            self.fail("Failed to move to center")
    
    def test_vacuum_control(self):
        try:
            self.dobot.suck(True)
            self.dobot.suck(False)
            self.assertTrue(True)
        except RuntimeError:
            self.fail("Failed to control vacuum pump")

if __name__ == "__main__":
    unittest.main()