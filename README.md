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


