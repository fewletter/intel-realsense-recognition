# -*- Overview -*-
The code in this github are based on many talented people,including
* pytorch_i3d.py, videotransform.py, train_i3d.py and train_elevator.py are based on https://github.com/piergiaj/pytorch-i3d
* pytorch_c3d.py is based on https://github.com/jfzhang95/pytorch-video-recognition
* videotransform.py is based on https://github.com/piergiaj/pytorch-i3d/blob/master/videotransforms.py
* save_bgremoved.py and oprealsense.py are based on https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples 

# -*- The Usage of Files -*-
"""
"""

#Code

* opnp.py
* opnp2.py
* opnp3.py
* opnp4.py
* opnp5.py
> The usage of the above files:
>> Show the results of save_bgremoved.py and other ....py file, or show the result of preporcessing image
* oprealsense.py
> The usage of oprealsense.py:
>> Open the Intel Realsense camera
* prepare_data2.py
> The usage of prepare_data2.py:
>> Prepare the training data and testing data, save the data title as csv file
* pytorch_c3d.py
> The usage of pytorch_c3d.py:
>> The structure of c3d model
* pytorch_i3d.py
> The usage of pytorch_i3d.py:
>> The structure of i3d model is based on "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
>>> ![image](https://user-images.githubusercontent.com/80392504/145832866-9b508fd2-9cb4-4ab6-a183-f53553dc5532.png) 
>>> The structure of I3D
>>> ![image](https://user-images.githubusercontent.com/80392504/145832942-c56ec63e-371c-4295-949c-96b1e754f82a.png) 
>>> The structure of Inception Module(Inc.)
* save_bgremoved.py
> The usage of save_bgremoved.py:
>> Transform depth raw file(16bit) to depth file(8bit) and save .bag files as .npy files
* train_i3d.py
* train_model_elevator.py
> The usage of the above files:
>> Training and testing
* videotransform.py
> The usage of videotransform.py
>> Transform the video,and the data augmentation of this training is RandomPadResizecrop
>>> The result of RandomPadResizecrop 


    class RandomPadResizecrop(object):
    def __init__(self):
        '''
        if data is contour,binary o_size = 640
        if use opnp2.py to open the npfile o_size = 1280
        '''
        self.o_size = 640
        self.size = 224
    
    def __call__(self,images):
        images = np.array(images)
        c,t,h,w = images.shape
        clips = []
        #紀錄手型座標極值
        height_high = []
        height_low = []
        width_left = []
        width_right= []
        if c == 1:
            images = np.transpose(images,[1,2,3,0])#clips[1,64,360,640] to clips[64,360,640,1]
            images = np.squeeze(images,axis=3)#clips[64,360,640] [t,h,w]
            '''
            run every frame,each frame size is (360,640),get bounding rectangle(x,y,x+w,y+h)
            '''
            for frame in range(t):
                ret,thresh = cv2.threshold(images[frame],10,255,0)#黑白影片所以閥值設10跟255
                contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cnts = sorted(contours, key=cv2.contourArea)
                for cnt in cnts:
                    if cv2.contourArea(cnt) < 600*300 and cv2.contourArea(cnt) > 60*60:
                        new_cnts = cnt
                        x,y,w,h = cv2.boundingRect(new_cnts)
                        height_high.append(y)
                        height_low.append(y+h)
                        width_left.append(x)
                        width_right.append(x+w) 
                
            #找出極值中的最大最小值
            x_left = min(width_left)
            x_right = max(width_right)
            y_low = min(height_high)
            y_high = max(height_low)
            
            #告訴圖片要pad多少
            pad_h = self.o_size-(y_high - y_low)
            pad_w = self.o_size-(x_right - x_left)
            part1 = random.randint(0,pad_h)
            part2 = random.randint(0,pad_w)
            
            #進行randompad
            #run every frame,get the randompad frame
            for clip in range(t):
                b = Image.fromarray(images[clip])
                b = b.crop((x_left, y_low, x_right, y_high))   
                pic = np.asarray(b)
                
                pic = np.pad(pic,((part1,pad_h-part1),(part2,pad_w-part2)),'constant')
                pic = Image.fromarray(pic)
                
                #進行resize
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
                
            clips = np.expand_dims(clips, axis=0)#clips sizes back to [1,64,360,640]
            
        '''
        if c == 3:
            a = np.pad(images,((0,0),(0,0),(int((w-h)/self.num)*part1,int((w-h)/self.num)*(self.num-part1)),(0,0)),'constant',constant_values=((0,0),(0,0),(160,160),(0,0)))
            images = np.transpose(a,[1,2,3,0])
            for frame in range(t):
                pic = Image.fromarray(images[frame])
                pic = pic.resize((224,224),Image.BICUBIC)
                pic = np.asarray(pic)
                clips.append(pic)
            clips = np.asarray(clips)
        '''
        return clips
        
    def __repr__(self):
        return self.__class__.__name__
>>>> ![image](https://user-images.githubusercontent.com/80392504/145834129-13674c9b-34f5-43e2-b346-bb4c76d31d51.png) first result
>>>> ![image](https://user-images.githubusercontent.com/80392504/145834159-44e56d5e-263b-4c9e-b21e-eb695730a52a.png) second result



# -*- The Dataset -*-
#Hand file

* bag file
    * What are in bag file?
         * There are both RGB video(8bit) and Depth video(16bit) in the bag file 
    * 20211109_174608.bag
    * 20211109_174618.bag
    * 20211109_174623.bag
    * 20211109_174628.bag
    * 20211109_174633.bag
    * 20211109_174638.bag
    * 20211109_174642.bag
    * ...
            
        
* npy file
    * What are in npy file?
        * 8bit_depth are frames which are preprocessed by some codes
        * depthraw are the depth frames which are saved in bag file 
    * THE MEANING OF THE NUMBER:
        * modality_subject_floor_trial 
    * 8bit_depth
        * 8bitdepth_1_1_1.npy
        * 8bitdepth_1_1_2.npy
        * 8bitdepth_1_1_3.npy
        * 8bitdepth_1_1_4.npy
        * 8bitdepth_1_1_5.npy
        * 8bitdepth_1_1_6.npy
        * 8bitdepth_1_1_7.npy
        * 8bitdepth_1_1_8.npy
        * ...
        * * video demo 
        > ![image](https://user-images.githubusercontent.com/80392504/145842462-4f762e3b-01f2-40ee-bbee-5a2af0e50cae.png)
        * image demo 
        > ![image](https://user-images.githubusercontent.com/80392504/145842939-037d3d44-ac9d-41d7-8ee7-87a4cdf1d75f.png)
    * depthraw
        * depthraw_1_1_1.npy
        * depthraw_1_1_2.npy
        * depthraw_1_1_3.npy
        * depthraw_1_1_4.npy
        * depthraw_1_1_5.npy
        * depthraw_1_1_6.npy
        * depthraw_1_1_7.npy
        * depthraw_1_1_8.npy
        * ...
        * video demo 
        > ![image](https://user-images.githubusercontent.com/80392504/145843852-f9949a82-a5c6-4b45-988a-99d0c4ba364e.png)
        * image demo 
        > ![image](https://user-images.githubusercontent.com/80392504/145843698-c4b477c0-dc27-41b3-b04a-4ba2117de523.png)


