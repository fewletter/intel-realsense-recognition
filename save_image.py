# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
import time

'''
設定gui圖形介面器

window = tk.Tk()
window.title('Realsense Stream')

main = tk.Frame(window, bg = 'white',height = 360, width = 640)
main.grid()
video = tk.Label(main)
video.grid()


window.geometry("640x640")

在anaconda3 prompt傳入參數
'''
# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
'''
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    break
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
'''
'''
存入檔案的函式
'''
print(args.input)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('l_rgb.mp4',fourcc, 15.0, (640,360))
out4 = cv2.VideoWriter('l_gray.mp4',fourcc, 15.0, (640,360))
out2 = cv2.VideoWriter('l_point.mp4',fourcc, 15.0, (640,360))
out3 = cv2.VideoWriter('l_gray_image.mp4',fourcc, 15.0, (640,360))
show_numpy = []
all_show_numpy = []
show_fpath = 'C:/Users/fewle/Documents/dataset/npy/test/show.npy'
show2_numpy = []
all_show2_numpy = []
show2_fpath = 'C:/Users/fewle/Documents/dataset/npy/test/show2.npy'
all_show3_numpy = []
show3_fpath = 'C:/Users/fewle/Documents/dataset/npy/test/show3.npy'

def slow_processing(frame):
    n = frame.get_frame_number() 
    if n % 1 == 0:
        time.sleep(1/8)
    print(n)
'''
開始讀bag檔
'''


try:
    # Create pipeline
            pipeline = rs.pipeline()

    # Create a config object
            config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            rs.config.enable_device_from_file(config,args.input) 
            #queue = rs.frame_queue(50, keep_frames=True)
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
            config.enable_stream(rs.stream.depth, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Start streaming from file
            profile = pipeline.start(config)
            #start = time.time()

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)

            clipping_distance_in_meters = 0.62 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

    # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
            align_to = rs.stream.color
            align = rs.align(align_to)
    # Create colorizer object
            colorizer = rs.colorizer()

    # Streaming loop
            while True:
        # Get frameset of depth
                frames = pipeline.wait_for_frames()
                #slow_processing(frames)
        # 對齊
                #aligned_frames = rs.composite_frame(frames)
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

        # remove background
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                gray_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
                
                depth_image_m = depth_image
                depth_image_m[np.where(depth_image_m==np.min(depth_image_m))] = np.max(depth_image_m) 
                distance = np.min(depth_image_m)
                new_distance = distance + 0.1/depth_scale
                
                gray_color = 160
                black_color = np.zeros((720,1280,3),np.uint8)
                white_color = np.ones((720,1280,3),np.uint8)*255
                black_color_show = np.zeros((720,1280,3),np.uint8)
                white_color_show = np.ones((720,1280,3),np.uint8)*255
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                gray_image_3d = np.dstack((gray_image,gray_image,gray_image))
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), gray_color, color_image)
                bg_removed_binary = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, white_color)
                
                bg_removed_gray_show = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color_show, gray_image_3d)
                bg_removed_point_show = np.where((depth_image_3d > distance + 0.002/depth_scale ) | (depth_image_3d <= distance), black_color_show, white_color_show)
                
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
        
                #color_image = rgb_to_bgr(color_image)
                bg_removed = rgb_to_bgr(bg_removed)
                
                color_image = cv2.resize(color_image.copy(),(640,360))
                bg_removed = cv2.resize(bg_removed.copy(),(640,360))
                depth_image = cv2.resize(depth_image.copy(),(640,360))
                depth_colormap = cv2.resize(depth_colormap.copy(),(640,360))
                bg_removed_binary = cv2.resize(bg_removed_binary.copy(),(640,360))
                bg_removed_gray_show = cv2.resize(bg_removed_gray_show.copy(),(640,360))
                bg_removed_point_show = cv2.resize(bg_removed_point_show.copy(),(640,360))
                gray_image_3d = cv2.resize(gray_image_3d.copy(),(640,360))
                
                show_numpy = np.copy(depth_image)
                show_numpy = np.expand_dims(show_numpy, axis = 0) 
                all_show_numpy.append(show_numpy) 
                
                show2_numpy = np.copy(bg_removed_binary)
                show2_numpy = np.expand_dims(show2_numpy, axis = 0) 
                all_show2_numpy.append(show2_numpy)
                
                show3_numpy = np.copy(color_image)
                show3_numpy = np.expand_dims(show3_numpy, axis = 0) 
                all_show3_numpy.append(show3_numpy)
                
        # Render image in opencv window
            # images = np.hstack((color_image, bg_removed))
            # cv2.imshow('Remove background',bg_removed)
            # cv2.imshow('Depth stream',depth_image)
                cv2.imshow('RGB stream',color_image)
                #cv2.imshow('Remove background',bg_removed)
                #cv2.imshow("normal stream",depth_colormap)
            # cv2.imshow('RGB and Depth images',images)
                key = cv2.waitKey(1)
            # if pressed escape exit program
        
        # 存成mp4檔
                
                out1.write(depth_colormap)
                out4.write(bg_removed_gray_show)
                out2.write(bg_removed_point_show)
                out3.write(gray_image_3d)
                

        # cv2.imshow("normal stream",depth_colormap)
        # out.write(depth_color_image)
        

        # 深度
        # print("shape of color image:{0}".format(depth_color_image.shape))
        # for i in range(848):
        #     for j in range(480):
        #         temp = depth_frame.get_distance(i, j)
        #         print(temp)
                if key == 27:
                    cv2.destroyAllWindows()

                    break
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    np.save(show_fpath, np.concatenate(all_show_numpy,axis=0))
                    np.save(show2_fpath, np.concatenate(all_show2_numpy,axis=0))
                    np.save(show3_fpath, np.concatenate(all_show3_numpy,axis=0))
                    break
finally:
       pass

'''
button = tk.Button(window, text = "輸入realsense攝像頭", width = 15, height = 2, command = stream())
button.place(x = 270, y = 500)
window.mainloop()

# 宣告攝影機



def stream():

    # 讀取當前的影像
    global status, frame
    status, frame = cap.read()
    # 如果有影像的話
    if status:
        # 將 OpenCV 色改格式 ( BGR ) 轉換成 RGB
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # 將 OpenCV 圖檔轉換成 PIL
        im_pil = Image.fromarray(im_rgb)
        # 轉換成 ImageTK
        imgTK = ImageTk.PhotoImage(image=im_pil)
        # 放入圖片
        div1.configure(image=imgTK)
        # 防止圖片丟失，做二次確認
        div1.image = imgTK
    # 10 豪秒 後執行 stream 函式，這裡是模擬 While 迴圈的部分
    window.after(10, stream) 

# 先執行一次
stream()

window.mainloop()

# 釋出攝影機記憶體、關閉所有視窗
cap.release()
cv2.destroyAllWindows()
'''