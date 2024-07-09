import pandas as pd
import matplotlib.pyplot as plt
import os
import WorldCoords as wc 

file_path = r'40777060\40777060_frames\lowres_wide.traj'

#read X and Y values of camera + list
x_positions = []
y_positions = []

with open(file_path, 'r') as file:
    for line in file:
        values = line.split()

        x_positions.append(float(values[-3]))
        y_positions.append(float(values[-2]))
   
#plot trajectory
plt.figure(figsize=(10, 7))
plt.plot(x_positions, y_positions, marker='s', ms=5, mfc='b',mec='b', color='c', label='Trajectory')
plt.plot(x_positions[0],y_positions[0],marker='*', ms='15', mec='r', mfc='r', label = 'Starting Point')

plt.plot(wc.x, wc.y, marker = 's',mfc='g',mec='g') #3d coords

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Camera Trajectory')
plt.legend()
plt.grid()
plt.show()
