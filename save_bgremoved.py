# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:22:56 2021

@author: fewle
"""

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
from  matplotlib import pyplot as plt
from PIL import Image
import time


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()


'''
開始讀bag檔
'''

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, args.input) 

# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

queue = rs.frame_queue(100, keep_frames=True)

# Start streaming from file
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = 0.7
clipping_distance = clipping_distance_in_meters / depth_scale

# Create opencv window to render image in
# cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
align_to = rs.stream.color
align = rs.align(align_to)
# Create colorizer object
colorizer = rs.colorizer()

class save_npfile:
    def __init__(self,all_bgremoved_numpy):
        all_bgremoved_numpy = []
        self.all_bgremoved_numpy = all_bgremoved_numpy
    def form_npfile(self,bgremoved_numpy):
        bgremoved_numpy_file = []
        bgremoved_numpy_file = np.copy(bgremoved_numpy)
        bgremoved_numpy_file = np.expand_dims(bgremoved_numpy_file, axis = 0) 
        self.all_bgremoved_numpy.append(bgremoved_numpy_file)
        #return self.all_bgremoved_numpy
    def save(self,folder,mid_folder_name,file_name):
        np.save(folder+mid_folder_name+file_name, np.concatenate(self.all_bgremoved_numpy,axis=0))

no_ebit_np = save_npfile([])
ebit_np = save_npfile([])
depth_np = save_npfile([])
smallest_list = []

def slower_processing(frame):
    n = frame.get_frame_number() 
    if n % 1 == 0:
        time.sleep(1/60)
    print(n)

start = time.time()

#profile.get_device().as_playback().set_real_time(True)

try:
    # Streaming loop
            while time.time() - start < 5:
        # Get frameset of depth
                frames = pipeline.wait_for_frames()
        # 對齊
                aligned_frames = align.process(frames)
                #slower_processing(frames)
                #m_frames = pipeline.wait_for_frames()
        # Get depth frame
                depth_frame = frames.get_depth_frame()
        # Get color frame
                color_frame = frames.get_color_frame()
                
                #print(frames.get_frame_number())
                
        # Convert depth_frame to numpy array to render image in opencv
                depth_image = np.asanyarray(depth_frame.get_data())
        # Convert color_frame to numpy array to render image in opencv
                color_image = np.asanyarray(color_frame.get_data())
                
                depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                distance = np.min(depth_image)
                print("========================================")
                print('The nearest distance is: '+ str(distance))

        # remove background
                '''
                8bit depth image
                '''
                #removed background
                smallest = np.min(depth_image)
                hand_range = 450
                test_range = 300
                
                if abs(2000-smallest) <= test_range:
                    print('After testing, the nearest distance is: '+ str(smallest))
                    smallest = smallest
                else:
                    depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                    smallest = np.min(depth_image)
                    if abs(2000-smallest) <= test_range:
                         print('After testing, the nearest distance is: '+ str(smallest))
                         smallest = smallest
                    else:
                        depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                        smallest = np.min(depth_image)
                        if abs(2000-smallest) <= test_range:
                            print('After testing, the nearest distance is: '+ str(smallest))
                            smallest = smallest
                        else:
                            depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                            smallest = np.min(depth_image)
                            if abs(2000-smallest) <= test_range:
                                print('After testing, the nearest distance is: '+ str(smallest))
                                smallest = smallest
                            else:
                                depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                                smallest = np.min(depth_image)
                                if abs(2000-smallest) <= test_range:
                                    print('After testing, the nearest distance is: '+ str(smallest))
                                    smallest = smallest
                                else:
                                    depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                                    smallest = np.min(depth_image)
                                    if abs(2000-smallest) <= test_range:
                                        print('After testing, the nearest distance is: '+ str(smallest))
                                        smallest = smallest
                                    else:
                                        depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                                        smallest = np.min(depth_image)
                                        if abs(2000-smallest) <= test_range:
                                            print('After testing, the nearest distance is: '+ str(smallest))
                                            smallest = smallest
                                        else:
                                            depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                                            smallest = np.min(depth_image)
                                            if abs(2000-smallest) <= test_range:
                                                print('After testing, the nearest distance is: '+ str(smallest))
                                                smallest = smallest
                                            else:
                                                depth_image[np.where(depth_image==np.min(depth_image))] = np.max(depth_image) 
                                                smallest = np.min(depth_image)
                                                print('After testing, the nearest distance is: '+ str(smallest))                   
                                    
                ebit_image = depth_image.copy()
                ebit_image[np.where(ebit_image > smallest+hand_range)] = 0
                #轉成8bit
                print('The farest distance is: ' + str(np.max(ebit_image)))
                #no_ebit_image = ebit_image.copy()
                ebit_image[np.where(ebit_image != 0)] = 255 - 140*(ebit_image[np.where(ebit_image != 0)]-smallest)/(hand_range) 
                ebit_image = np.asarray(ebit_image,np.uint8)
                
        # 找出圖片大小 
                def frame_shape(image):
                    width = int(image.shape[0])
                    height = int(image.shape[1])
                    dimensions = (width,height)
                    return dimensions
                
        # 轉換rgb到bgr
                def rgb_to_bgr(image):
                    r,g,b = cv2.split(image)
                    new_image =  cv2.merge([b,g,r])
                    return new_image
                
                color_image = rgb_to_bgr(color_image)

        # 縮小圖片        
                depth_image = cv2.resize(depth_image.copy(),(640,360))
                ebit_image = cv2.resize(ebit_image.copy(),(640,360))
        
        #draw contour
                '''     
                gray = cv2.cvtColor(bg_removed_binary_show.copy(),cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 10)        
                ret,thresh = cv2.threshold(blur,10,250,0)#黑白影片所以閥值設10跟255
                contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                #bg_removed_rgb = cv2.drawContours(bg_removed_rgb.copy(), contours, -1, (0,0,255), 3)
                cnts = sorted(contours, key=cv2.contourArea)
                dis = []
                for cnt in cnts:
                    if cv2.contourArea(cnt) < 600*300 and cv2.contourArea(cnt) > 60*60:
                        new_cnts = cnt
                        x,y,w,h = cv2.boundingRect(new_cnts)
                        cv2.drawContours(black_color_show, new_cnts, -1, (255,255,255), 5)
                '''
                
        #save bg_removed file
                ebit_np.form_npfile(ebit_image)
                depth_np.form_npfile(depth_image)
                #no_ebit_np.form_npfile(no_ebit_image)
        # Render image in opencv window
                cv2.imshow('8bit stream',ebit_image)
                key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()

                    break
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    ebit_np.save('C:/Users/fewle/Documents/dataset/npy/','test/','depthc_test.npy')
                    #depth_np.save('C:/Users/fewle/Documents/dataset/npy/','l_depth/','depthraw_1_b2_10.npy')
                    #no_ebit_np.save('C:/Users/fewle/Documents/dataset/npy/','test/','no8bit_test.npy')
                    break
finally:
        pipeline.stop()
