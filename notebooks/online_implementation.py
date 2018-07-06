# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:56:41 2018

@author: Monkey-PC
"""

import keyboard
import numpy as np
from pylsl import StreamInlet, resolve_stream, local_clock
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt1

from filtering_cca import *

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


    




plot_duration = 5.0


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
# create a new inlet to read from the stream


# Create the pyqtgraph window
win = pg.GraphicsWindow()
# win.setWindowTitle('LSL Plot ' + inlet.info().name())
plt = win.addPlot()
# plt2 = win.addPlot() 
# plt.setLimits(xMin=0.0, xMax=plot_duration, yMin=-1.0 * (inlet.channel_count - 1), yMax=1.0)
# plt2.setLimits(xMin=0.0, xMax=plot_duration, yMin=-1.0 * (inlet.channel_count - 1), yMax=1.0)
t0 = [local_clock()] * inlet.channel_count
curves = []
# curves += [plt2.plot()]
for ch_ix in range(inlet.channel_count):
    curves += [plt.plot()]
#     print(curves)



def update():
    global inlet, curves, t0
    # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=32)
    
    if timestamps:
        timestamps = np.asarray(timestamps)
        y = np.asarray(chunk)
        #print(y.shape)
        for ch_ix in range(inlet.channel_count):
            #print(ch_ix)
            old_x, old_y = curves[ch_ix].getData()
            if old_x is not None:
                old_x += t0[ch_ix]  # Undo t0 subtraction
                this_x = np.hstack((old_x, timestamps))
                this_y = np.hstack((old_y, y[:, ch_ix] ))
            else:
                this_x = timestamps
                this_y = y[:, ch_ix] 
#             print(this_y.shape)    
            t0[ch_ix] = this_x[-1] - plot_duration
            this_x -= t0[ch_ix]
            b_keep = this_x >=0
            curves[ch_ix].setData(this_x[b_keep], this_y[b_keep])
            


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


# # Start Qt event loop unless running in interactive mode or using pyside.
# if name == 'main':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()


# Start Qt event loop unless running in interactive mode or using pyside.
if name == 'main':
    import sys
    first_flag =1
    t = 1;
    sampling_rate=128;
    y = np.array([]);
    t_ = np.array([]);
    threshold = 0.4
    r_past = threshold*np.ones((1,20))
    counter = 0
    while(1):
        update()
        if keyboard.is_pressed('A'):
            break
        t = t+1
        if (t > 200000):
            X_input = np.array([i.getData()[1] for i in curves])
            X_input = X_input.T;
            r = cca_corrs(X_input, 15, 12, 8, sampling_rate)
            print(r)
            r_past(counter) = np.max(r)
            if counter == 19:
                counter = 0
            else:
                counter = counter +1
            threshold = np.median(r_past)*0.7
            if np.max(r) > threshold:
                y_est = np.argmax(r) + 2
            else:
                y_est = 1
            y = np.append(y, y_est)
            t_ = np.append(t_,t)
            y = y.flatten();
            t_ = t_.flatten();
            print(y_est)
            move_robot(y_est)
    plt1.plot(t_,y-1)
    plt1.show()