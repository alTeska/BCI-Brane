import time
import numpy as np
import urllib
import urllib.request  # this library is to access url
from pynput import keyboard  # access keyboard
from pynput.keyboard import Key, Listener

IP = 'http://192.168.0.154/arduino/mode'
speedRight = 0
speedLeft = 0
moveOption = 'forward'

def send_url(IP, moveOption, speedLeft, speedRight):
    urlAddress = np.array([IP, moveOption, speedLeft, speedRight])
    urlAddress = '/'.join(i for i in urlAddress)

    urllib.request.urlopen(urlAddress)


def on_press(key):
    if key == Key.esc:
        send_url(IP, 'forward', 0, 0)
        return False

    if key == Key.up:
        send_url(IP, 'forward', 200, 200)

    if key == Key.down:
        send_url(IP, 'forward', -200, -200)

    if key == Key.left:
        send_url(IP, 'forward', -200, 200)

    if key == Key.right:
        send_url(IP, 'forward', 200, -200)

# send_url(IP, moveOption, speedRight, speedLeft)

def on_release(key):

    moveOption = 'forward'
    speedRight = 0
    speedLeft = 0

    send_url(IP, moveOption, speedLeft, speedRight)



# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
