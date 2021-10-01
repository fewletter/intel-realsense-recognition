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

# Start streaming from file
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = 0.69
clipping_distance = clipping_distance_in_meters / depth_scale

# Create opencv window to render image in
# cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
align_to = rs.stream.color
align = rs.align(align_to)
# Create colorizer object
colorizer = rs.colorizer()

string1 = 'C:/Users/fewle/Documents/dataset/npy/'
string2 = '3_b2_9.npy'

bgremoved_binary_numpy = []
all_bgremoved_binary_numpy = []
bg_removed_fpath = string1+'contour/contour_'+string2

bgremoved_rgb_numpy = []
all_bgremoved_rgb_numpy = []
bg_removed_rgb_fpath = string1+'l_rgb360/rgb_'+string2

bgremoved_b_numpy = []
all_bgremoved_b_numpy = []
bg_removed_b_fpath = string1+'l_binary/binary_'+string2

bgremoved_point_numpy = []
all_bgremoved_point_numpy = []
bg_removed_point_fpath = string1+'l_point/point_'+string2

bgremoved_gray_numpy = []
all_bgremoved_gray_numpy = []
bg_removed_gray_fpath = string1+'l_gray/gray_'+string2


try:
    # Streaming loop
            while True:
        # Get frameset of depth
                frames = pipeline.wait_for_frames()
        # 對齊
                aligned_frames = align.process(frames)
        # Get depth frame
                depth_frame = aligned_frames.get_depth_frame()
        # Get color frame
                color_frame = aligned_frames.get_color_frame()
        

        # Colorize depth frame to jet colormap
        # depth_color_frame = colorizer.colorize(depth_frame)


        # Convert depth_frame to numpy array to render image in opencv
                depth_image = np.asanyarray(depth_frame.get_data())
        # Convert color_frame to numpy array to render image in opencv
                color_image = np.asanyarray(color_frame.get_data())
                
                depth_image_m = depth_image
                #print(np.where(depth_image_m==np.min(depth_image_m)))
                
                depth_image_m[np.where(depth_image_m==np.min(depth_image_m))] = np.max(depth_image_m) 
                distance = np.min(depth_image_m)
                new_distance = distance + 0.1/depth_scale
                
                #print(np.argmin(depth_image_m))
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                gray_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        # remove background
                grey_color = np.ones((720,1280,3),np.uint8)*160
                black_color = np.zeros((720,1280),np.uint8)
                white_color = np.ones((720,1280),np.uint8)*255
                black_color_show = np.zeros((720,1280,3),np.uint8)
                white_color_show = np.ones((720,1280,3),np.uint8)*255
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #rgb所以需開三個通道
                gray_image_3d = np.dstack((gray_image,gray_image,gray_image))
                '''
                RGB
                '''
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
                '''
                GRAY
                '''
                bg_removed_gray = np.where((depth_image > clipping_distance) | (depth_image <= 0), black_color, gray_image)
                bg_removed_gray_show = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color_show, gray_image_3d)
                '''
                BINARY
                '''
                bg_removed_binary = np.where((depth_image > clipping_distance) | (depth_image <= 0), black_color, white_color)
                bg_removed_binary_show = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color_show, white_color_show)
                '''
                POINT
                '''
                bg_removed_point = np.where((depth_image > distance + 0.004/depth_scale ) | (depth_image <= distance), black_color, white_color)
                bg_removed_point_show = np.where((depth_image_3d > distance + 0.004/depth_scale ) | (depth_image_3d <= distance), black_color_show, white_color_show)
                
        
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
                color_image = cv2.resize(color_image.copy(),(640,360))
                #depth_image = cv2.resize(depth_image.copy(),(640,360))
                #depth_colormapr = cv2.resize(depth_colormap.copy(),(640,360))
                #bg_removed_binary = cv2.resize(bg_removed_binary.copy(),(640,360))
                #bg_removed_depth = cv2.resize(bg_removed_depth.copy(),(640,360))
                depth_image = cv2.resize(depth_image.copy(),(640,360))
                bg_removed_show = cv2.resize(bg_removed_binary_show.copy(),(640,360))
                bg_removed_binary_show = cv2.resize(bg_removed_binary_show.copy(),(640,360))
                black_color_show = cv2.resize(black_color_show.copy(),(640,360))
                black_color = cv2.resize(black_color.copy(),(640,360))
                bg_removed_gray_show = cv2.resize(bg_removed_gray_show.copy(),(640,360))
                bg_removed_binary = cv2.resize(bg_removed_binary.copy(),(640,360))
                bg_removed_point = cv2.resize(bg_removed_point.copy(),(640,360))
                bg_removed_point_show = cv2.resize(bg_removed_point_show.copy(),(640,360))
                bg_removed_gray = cv2.resize(bg_removed_gray.copy(),(640,360))

        
        #padding       
                #pad_bg_removed_binary = cv2.copyMakeBorder(bg_removed_binary_show.copy(),int((640-360)/2),int((640-360)/2),0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                #pad_black_color_show = cv2.copyMakeBorder(black_color_show,int((640-360)/2),int((640-360)/2),0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        #draw contour        
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
                
                #gray_2 = cv2.cvtColor(bg_removed_binary.copy(),cv2.COLOR_BGR2GRAY)
                blur_2 = cv2.GaussianBlur(bg_removed_binary.copy(), (5,5), 10)        
                ret_2,thresh_2 = cv2.threshold(blur_2,10,250,0)#黑白影片所以閥值設10跟255
                contours_2,hierachy_2 = cv2.findContours(thresh_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                #bg_removed_rgb = cv2.drawContours(bg_removed_rgb.copy(), contours, -1, (0,0,255), 3)
                cnts_2 = sorted(contours_2, key=cv2.contourArea)
                dis = []
                for cnt in cnts_2:
                    if cv2.contourArea(cnt) < 600*300 and cv2.contourArea(cnt) > 60*60:
                        new_cnts_2 = cnt
                        x,y,w,h = cv2.boundingRect(new_cnts_2)
                        cv2.drawContours(black_color, new_cnts_2, -1, (255,255,255), 5)
                        
        #save bg_removed file
                bgremoved_binary_numpy = np.copy(black_color)
                bgremoved_binary_numpy = np.expand_dims(bgremoved_binary_numpy, axis = 0) 
                all_bgremoved_binary_numpy.append(bgremoved_binary_numpy)       
                
                bgremoved_rgb_numpy = np.copy(bg_removed)
                bgremoved_rgb_numpy = np.expand_dims(bgremoved_rgb_numpy, axis = 0) 
                all_bgremoved_rgb_numpy.append(bgremoved_rgb_numpy) 
                
                bgremoved_b_numpy = np.copy(bg_removed_binary)
                bgremoved_b_numpy = np.expand_dims(bgremoved_b_numpy, axis = 0) 
                all_bgremoved_b_numpy.append(bgremoved_b_numpy)
                
                bgremoved_point_numpy = np.copy(bg_removed_point)
                bgremoved_point_numpy = np.expand_dims(bgremoved_point_numpy, axis = 0) 
                all_bgremoved_point_numpy.append(bgremoved_point_numpy)
                
                bgremoved_gray_numpy = np.copy(bg_removed_gray)
                bgremoved_gray_numpy = np.expand_dims(bgremoved_gray_numpy, axis = 0) 
                all_bgremoved_gray_numpy.append(bgremoved_gray_numpy)
        # Render image in opencv window
                #images = np.hstack((depth_colormapr))
                #cv2.imshow('Remove background',bg_removed_binary)
                #cv2.imshow('Remove background Depth Image',bg_removed_depth)
                cv2.imshow('Remove background RGB',bg_removed_binary_show)
                cv2.imshow('Depth stream',bg_removed_point_show)
                cv2.imshow('result',bg_removed_gray_show)
                #cv2.imshow('Inpaint image',bg_inpaint)
                #cv2.imshow('bg_removed dilate',bg_dilate)
                cv2.imshow('RGB stream',color_image)
                #cv2.imshow('bg_removed and rgb_image',)
                #cv2.imshow('depth_colormap',depth_colormapr)
                key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()

                    break
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    np.save(bg_removed_fpath, np.concatenate(all_bgremoved_binary_numpy,axis=0))
                    np.save(bg_removed_rgb_fpath, np.concatenate(all_bgremoved_rgb_numpy,axis=0))
                    np.save(bg_removed_b_fpath, np.concatenate(all_bgremoved_b_numpy,axis=0))
                    #np.save(bg_removed_point_fpath, np.concatenate(all_bgremoved_point_numpy,axis=0))
                    np.save(bg_removed_gray_fpath, np.concatenate(all_bgremoved_gray_numpy,axis=0))
                    break
finally:
       pipeline.stop()
