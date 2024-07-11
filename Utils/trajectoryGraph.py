import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
import Utils.WorldCoords as wc

#r'40777060/40777060_frames/lowres_wide.traj
# file_path = r'40777060\40777060_frames\lowres_wide.traj'

def graphTraj(x,y, file_path, imgname):
    print(file_path.split("/")[1])
    print(imgname)
    wx, wy = wc.load_img_info(file_path.split("/")[1], imgname)

    if (wx == None or wy == None):
        print("Error: Could not load image info")
        st.write('Confidence too low to find exact location.')

    

        
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
    plt.plot(x, y,marker='*', ms='15', mec='r', mfc='r', label = 'Position of camera')
    if (wx != None and wy != None):
        plt.plot(wx, wy, marker = 's',mfc='g',mec='g', label= 'object location') #3d coords
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Camera Trajectory')
    plt.legend()
    plt.grid()
    st.pyplot(plt)