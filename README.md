# -*- The Usage of Files -*-
"""
"""

#Code

* opnp.py
* opnp2.py
* opnp3.py
* opnp4.py
* opnp5.py
    * The usage of the above files:
        * show the results of save_bgremoved.py or other ....py file
        * some files are used to do image processing
* oprealsense.py
    * The usage of oprealsense.py:
        * open the Intel Realsense camera
* prepare_data2.py
    * The usage of prepare_data2.py:
        * prepare the training data and testing data, save the data title as csv file
* pytorch_c3d.py
    * The usage of pytorch_c3d.py:
        * The structure of c3d model
* pytorch_i3d.py
    * The usage of pytorch_i3d.py:
        * The structure of i3d model
* save_bgremoved.py
    * The usage of save_bgremoved.py:
        * transform depth raw file(16bit) to depth file(8bit)
        * save .bag files as .npy files
* train_i3d.py
* train_model_elevator.py
    * The usage of the above files:
        * Training and testing
* videotransform.py
    * The usage of videotransform.py
        * transform the video, in the training I use RandomPadResizecrop

# -*- The Dataset -*-
#Hand file

* bag file
    * 20211109_174608.bag
    * 20211109_174618.bag
    * 20211109_174623.bag
    * 20211109_174628.bag
    * 20211109_174633.bag
    * 20211109_174638.bag
    * 20211109_174642.bag
    * ...
            
        
* npy file
    * THE MEANING OF THE NUMBER:
        * The form of this video_The number of person_The number of the floor_How many times has this floor be filmed 
    * 8bit_depth
        * depth_1_1_1.npy
        * depth_1_1_2.npy
        * depth_1_1_3.npy
        * depth_1_1_4.npy
        * depth_1_1_5.npy
        * depth_1_1_6.npy
        * depth_1_1_7.npy
        * depth_1_1_8.npy
        * ...
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
