

import sys
import os
import cv2
import numpy as np 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  

        self.skin_detection_flag = False 
        self.face_detection_flag = False
        self.edge_detection_flag = False
        self.connected_components_flag = False
        self.iron_man_mask_flag = False

        self.cap = cv2.VideoCapture()  
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()  
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  


        #‎Declare Button
        self.button_open_camera = QtWidgets.QPushButton(u'Open Camera')
        self.button_skin_detect = QtWidgets.QPushButton(u'Skin Detection')
        self.button_edge_detect = QtWidgets.QPushButton(u'Edge Detection')
        self.button_connected_components = QtWidgets.QPushButton(u'Connected Components')
        self.button_face_detect = QtWidgets.QPushButton(u'Face Detection')
        self.button_iron_man = QtWidgets.QPushButton(u'Iron Man')



        self.button_close = QtWidgets.QPushButton(u'Exit')

        #‎Set Button's Height
        self.button_open_camera.setMinimumHeight(50)
        self.button_skin_detect.setMinimumHeight(50)
        self.button_edge_detect.setMinimumHeight(50)
        self.button_connected_components.setMinimumHeight(50)
        self.button_face_detect.setMinimumHeight(50)
        self.button_iron_man.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        #Add Button in layout
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_skin_detect)
        self.__layout_fun_button.addWidget(self.button_edge_detect)
        self.__layout_fun_button.addWidget(self.button_connected_components)
        self.__layout_fun_button.addWidget(self.button_face_detect)
        self.__layout_fun_button.addWidget(self.button_iron_man)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'OpenCV')


    def slot_init(self):  # for bottom connect
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)  
        # The effects below use the same method to add
        # ex: button_skin_detect_click --> set skin detection flag =True --> show effect   
        self.button_skin_detect.clicked.connect(self.button_skin_detect_click)
        self.button_face_detect.clicked.connect(self.button_face_detect_click)
        self.button_edge_detect.clicked.connect(self.button_edge_detect_click)
        self.button_connected_components.clicked.connect(self.button_connected_components_click)

        self.button_iron_man.clicked.connect(self.button_iron_man_click)


        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        # print('self.timer_camera.isActive()',self.timer_camera.isActive())
        if self.timer_camera.isActive() == False:
            
            flag = self.cap.open(self.CAM_NUM)
            
            if flag == False:

                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'請檢測相機與電腦是否連接正確',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:

                self.timer_camera.start(30)
                self.button_open_camera.setText(u'Close Camera')
        else:
            
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'Open Camera')

    def button_skin_detect_click(self):

        if self.skin_detection_flag :

            self.timer_camera.timeout.connect(self.show_camera)
            self.button_skin_detect.setText(u'Skin Detection')
            self.skin_detection_flag = False

        else:

            self.timer_camera.timeout.connect(self.show_effect)
            self.button_skin_detect.setText(u'Close')
            self.skin_detection_flag = True
    
    def button_edge_detect_click(self):

        if self.edge_detection_flag :

            self.timer_camera.timeout.connect(self.show_camera)
            self.button_edge_detect.setText(u'Edge Detection')
            self.edge_detection_flag = False

        else:

            self.timer_camera.timeout.connect(self.show_effect)
            self.button_edge_detect.setText(u'Close')
            self.edge_detection_flag = True
    
    def button_connected_components_click(self):

        if self.connected_components_flag :
            self.timer_camera.timeout.connect(self.show_camera)
            self.button_connected_components.setText(u'Connected Components')
            self.connected_components_flag = False

        else:

            self.timer_camera.timeout.connect(self.show_effect)
            self.button_connected_components.setText(u'Close')
            self.connected_components_flag = True
      
    def button_face_detect_click(self):

        if self.face_detection_flag :
           
            self.timer_camera.timeout.connect(self.show_camera)
            self.button_face_detect.setText(u'Face Detection')
            self.face_detection_flag = False
        
        else:
            
            self.timer_camera.timeout.connect(self.show_effect)
            self.button_face_detect.setText(u'Close')
            self.face_detection_flag = True

    def button_iron_man_click(self):

        if  self.iron_man_mask_flag :
            self.timer_camera.timeout.connect(self.show_camera)
            self.button_iron_man.setText(u'Iron Man')
            self.iron_man_mask_flag = False
        else:
            self.timer_camera.timeout.connect(self.show_effect)
            self.button_iron_man.setText(u'Close')
            self.iron_man_mask_flag=True

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # Human Skin Detection Using HSV
    def skin_detection(self,image):
        # Condition：0<=H<=20，S>=58，V>=30
        low = np.array([0, 58, 30]) #Lower bound [H,S,V]
        high = np.array([20, 255, 255]) #Upper bound [H,S,V]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Convert RGB to HSV
        mask = cv2.inRange(hsv,low,high) 
        skin = cv2.bitwise_and(image,image, mask = mask) 
        cv2.putText(skin,'skin detection',(0,50), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_8)

        return skin
    
    # Edge Detection Using Canny Edge Detector 
    def edge_effect(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel_size = 3
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
        low_threshold = 10
        high_threshold = 100
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        
        if self.edge_detection_flag:
            cv2.putText(edges,'edge detection',(0,50), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_8)
            edges  = cv2.cvtColor(edges ,cv2.COLOR_GRAY2RGB)
            return edges
        
        if self.connected_components_flag:
            
            # img = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
            
            num_labels, labels_im = cv2.connectedComponents(edges)
            # Map component labels to hue val
            label_hue = np.uint8(179*labels_im/np.max(labels_im))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            
            # cvt to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            # set bg label to black
            labeled_img[label_hue==0] = 0
            cv2.putText(labeled_img,'connected components',(0,50), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_8)

            return labeled_img

    #Overlay PNG image with alpha channel
    def transparent_png_overlay(self,roi, overlay):

        h, w, _ = overlay.shape  # Size of mask
        rows, cols, _ = roi.shape  # Size of ROI
        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if i >= rows or j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0)  # get the alpha channel
                roi[i][j] = alpha * overlay[i][j][:3] + (1 - alpha) * roi[i][j]
        return roi 
  
    # Face Detection Using Haar Feature-based Cascade Classifier
    def face_effect(self,image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 
                                        scaleFactor=1.2, 
                                        minNeighbors=3, 
                                        minSize=(100, 100),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
        for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                
                if self.face_detection_flag:
                    cv2.putText(image,'face detection',(0,50), cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),2,cv2.LINE_8)
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 5)
                
                if self.iron_man_mask_flag:
                    mask = cv2.imread('ironman.png', -1)
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LANCZOS4)            
                    cv2.putText(image,'I am iron man',(x,y-50), cv2.FONT_HERSHEY_DUPLEX, 1,(20,39,175),2,cv2.LINE_8)
                    self.transparent_png_overlay(face_roi, mask)

        return image

    # Add any new effect with the flag
    def show_effect(self):
        
        flag, self.image = self.cap.read()
        
        show = cv2.resize(self.image, (640, 480))

        if self.skin_detection_flag:
            show = self.skin_detection(show)
        elif self.edge_detection_flag:
            show = self.edge_effect(show)
        elif self.connected_components_flag:
            show = self.edge_effect(show)
        elif self.face_detection_flag:
            show = self.face_effect(show)
        elif self.iron_man_mask_flag:
            show = self.face_effect(show)
        
       
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)

        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'Exit', u'Do you wanna exit！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        
        ok.setText(u'Sure')
        cancel.setText(u'Cancel')

        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':

    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())



