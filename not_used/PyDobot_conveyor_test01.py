from serial.tools import list_ports
import time

from not_used.dobot_extensions import Dobot

port = list_ports.comports()[0].device
port = 'COM18'
device = Dobot(port=port)

while 1<2:
    device.conveyor_belt_distance(20, 10, 1, 0)
    time.sleep(2)
    device.conveyor_belt_distance(0, 0, 0, 0)
 

# for i in range(500000):
#     print(i)
#     device.conveyor_belt_distance(10, 150, 1, 0)
    
    # if i % 2 == 3:
    #     print("even")
    #     device.conveyor_belt_distance(10, 150, -1, 0)
    # else:
    #     print("odd")
    #     device.conveyor_belt_distance(10, 5, 1, 0)
    # #time.sleep(0.5)

device.close()