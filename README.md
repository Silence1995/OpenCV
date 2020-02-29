# OpenCV

### Project
Using OpeCV to build some computer vision application, and using PyQt5 to build GUI 

### Requirements

- OpenCV
- PyQt5
- Numpy

### Application

- Skin Color Detection
  - Setting HSV value to detect the skin color range
- Face Detection
  - Using Haar cascades classifier
  - [Pre-trained classifier xml file](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- Edge Detection
  - Using Canny edge detector
- Connected Components
  - To detect connected regions in binary image
- Mask Effect 
  - Blending two images 


|       GUI      |    Skin Detection    |
|:--------------:|:--------------------:|
|  <img src="https://github.com/Silence1995/OpenCV/blob/master/results/GUI.PNG" width="350" height="150" />        |                      <img src="https://github.com/Silence1995/OpenCV/blob/master/results/skin_detection.PNG" width="350" height="150" />   |
| Edge Detection | Connected Components |
|       <img src="https://github.com/Silence1995/OpenCV/blob/master/results/edge_detection.PNG" width="350" height="150" />          |              <img src="https://github.com/Silence1995/OpenCV/blob/master/results/connected_components.PNG" width="350" height="150" />         |
| Face Detection |      Mask Effect     |
|        <img src="https://github.com/Silence1995/OpenCV/blob/master/results/face_detection.PNG" width="350" height="150" />         |              <img src="https://github.com/Silence1995/OpenCV/blob/master/results/mask.PNG" width="350" height="150" />        |


### Reference
- [PyQt5 & OpenCV](https://www.twblogs.net/a/5c55d115bd9eee06ee21b390)
- [Connected Component Labeling in python](https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python)
- [Overlay image python OpenCV](https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv)
- [Skin detection techniques](https://nalinc.github.io/blog/2018/skin-detection-python-opencv/)
