# intel-realsense-recognition
We use intelrealsense l515 to capture both rgb and depth frames


![image](https://user-images.githubusercontent.com/80392504/135630176-c5d7707f-fdc4-4829-9fa4-b7387a1c7f56.png) RGB frames
![image](https://user-images.githubusercontent.com/80392504/135630229-6f171837-4905-41a4-a352-049fe5fa7216.png) Depth frames
![image](https://user-images.githubusercontent.com/80392504/135630384-850154b3-b59a-4804-94ae-de832e4adb89.png) binary hand

We use i3d model to train our data, and you can find the model structure in pytorch_i3d

The opnp2.py file can see the histogram of the depth pixel distribution in depth frames

![image](https://user-images.githubusercontent.com/80392504/135631744-74d72467-75e9-41e0-a3dc-2158d5c7ef44.png)

In the videotransform.py file, there is a new data augment way which I called randompad
here is how it works:

![image](https://user-images.githubusercontent.com/80392504/135632481-bf802f2d-b472-4b9c-a7f8-53f36b9abfc9.png)
step 1 
![image](https://user-images.githubusercontent.com/80392504/135632521-918d8913-4327-4331-b52b-812350be2802.png)
step 2 
![image](https://user-images.githubusercontent.com/80392504/135632548-b69893f2-741c-4508-8d08-de0ec78f832a.png)
![image](https://user-images.githubusercontent.com/80392504/135632570-33519316-3545-4b23-a293-0cd508aa96e6.png)
final result(the place where hands are won't be the same every times) 
