import numpy as np
import urllib
import urllib.request  # this library is to access url

IP = 'http://192.168.0.184/arduino/mode'

def send_url(IP, moveOption= 'forward', speedLeft=0, speedRight=0):
    urlAddress = np.array([IP, moveOption, speedLeft, speedRight])
    urlAddress = '/'.join(i for i in urlAddress)
    urllib.request.urlopen(urlAddress)

def move_robot(y_est):
    if y_est == 1:
        send_url(IP, 'forward', 0, 0)

    if y_est == 2:
        send_url(IP, 'forward', 100, 100)

    if y_est == 3:
        send_url(IP, 'forward', -100, 100)

    if y_est == 4:
        send_url(IP, 'forward', 100, -100)

while(True):
    y_est = np.random.randint(4)
    move_robot(y_est)
