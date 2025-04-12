from dobot_control import DobotControl
import time

def main():
    """
    Example script demonstrating object sorting with DobotControl.
    Requires DobotDll.dll and dependencies (msvcp120.dll, msvcr120.dll, Qt5Core.dll,
    Qt5Network.dll, Qt5SerialPort.dll) in the same directory or system PATH.
    """
    dobot = DobotControl("DobotDll.dll")
    
    try:
        if not dobot.connect():
            print("Failed to connect to Dobot")
            return
        
        print("Moving to home")
        dobot.home()
        time.sleep(1)
        
        print("Moving to pick position (180, 0, 0)")
        dobot.move_to(180, 0, 0, 0)
        dobot.vacuum_on()
        time.sleep(0.5)
        
        print("Moving to place position (150, 147, 0)")
        dobot.move_to(150, 147, 0, 0)
        dobot.vacuum_off()
        time.sleep(0.5)
        
        print("Returning to home")
        dobot.home()
        
    finally:
        dobot.disconnect()

if __name__ == "__main__":
    main()
