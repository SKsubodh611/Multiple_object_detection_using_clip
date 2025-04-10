OpenCV installation does not have the ximgproc module, which contains the createSelectiveSearchSegmentation() function.


pip install opencv-contrib-python==4.8.1.78

pip install numpy==1.26.4


since it is high computational load ,as it uses selective search with clip 
for that change extract frame from 50 to 10 or 15 . i kept it 15

clip+selective search is very slow that's why ,
using ss. switchtoselectivesearchfast from ss.quality to make it faster

#test4.py
using  blip bootstrapped language image pretrained 

roboflow 
