import cv2 as cv
import numpy as np
import sys
import os.path
import random
import math
import json
from PIL import ImageFont, ImageDraw, Image
import pyautogui
from pathlib import Path
import mediapipe as mp

# modelName = "d3/r_7236_0430_open_lips_l_lab.h5" #左上角偶爾fa upper lips x
modelName = "init.h5" #no fa, 0.0x, all pass upper lips only middle 3
# modelName = "d3/n_7224_0430_open_lips_l_lab.h5" #no fa, normal stuck at 0.4x, all pass upper lips x
# modelName = "d3/nm_7272_0430_open_lips_l_lab.h5" #no fa, open /8, upper and smile /5

LEFT_BROW = [54,193]
RIGHT_BROW = [151,353]
BROW = [54, 353]
NOSE = [119,426]
LIPS = [207,430]


'''
INITIALIZATION
'''

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


# ref_file = open("ref.txt","r")
# modelName = ref_file.readline()



cap = cv.VideoCapture(0)


fontpath = "NotoSansTC-Medium.otf"



 
s_size = pyautogui.size()
width = s_size[0]
height = s_size[1]
cv.namedWindow('subtle facial', cv.WINDOW_NORMAL)
cv.resizeWindow('subtle facial', width, height)
cv.moveWindow('subtle facial', 0, 0)


prompt = True
response1 = False


pressed1 = False
pressed2 = False
show_control = False
show_instru = False
rest = 0




def draw_circle_red(event,x,y,flags,param):
    global response1
    global response2
    global pressed1
    global pressed2
    global show_control
    global show_instru
    global rest
    if prompt and event == cv.EVENT_LBUTTONDOWN and not response1:
        response1 = True
        response2 = False
    elif prompt and event == cv.EVENT_LBUTTONDOWN and response1:
        response1 = False
        response2 = True
    elif event == cv.EVENT_LBUTTONDOWN and not pressed1:
        pressed1 = True
        show_control = True
    elif event == cv.EVENT_LBUTTONDOWN and pressed1:
        pressed1 = False
        show_instru = False
        pressed2 = True

cv.setMouseCallback('subtle facial',draw_circle_red)
 

        


# calculate positions for targets////////////////////////////
mid_pt = []
for y in range(int(height/4),height,int(height/4)):
    for x in range(int(width/4),width,int(width/4)):
        mid_pt.append([x,y])

pt_pos = random.randint(0,8) 
pt_cnt = np.zeros(9)

        
        
'''
CNN MODEL
'''        

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


'''
TESTING PROMPT
'''

prompt = True


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    #cv.putText(frame, "END ANALYSIS. TO START TESTING, PRESS V", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    font = ImageFont.truetype(fontpath, 35)      
    imgPil = Image.fromarray(frame)                
    draw = ImageDraw.Draw(imgPil)                
    draw.text((10, 10), "結束分析，接下來將進行測試\n請看著畫面中的綠點做出微表情\n若有順利偵測請按y鍵，否則按n鍵\n準備好請點擊畫面繼續", fill=(0, 255, 0), font=font)
    frame = np.array(imgPil)
    cv.imshow("subtle facial", frame)
    if response1:
        break
    key = cv.waitKey(1)
    if key == ord('q'):
        sys.exit(0)



'''
TESTING
'''

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

model = load_model(modelName)


prompt = False
pt_pos = 0
correct = 0
success_cnt = 0
with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
    while (True):
        ret, frame = cap.read()
        if not ret:
            break

        # detectfaces
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
#         results = face_detector.process(rgb_frame)
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            continue;
#         if results.detections:
#             face_react = np.multiply(
#         [
#             results.detections[0].location_data.relative_bounding_box.xmin,
#             results.detections[0].location_data.relative_bounding_box.ymin,
#             results.detections[0].location_data.relative_bounding_box.width,
#             results.detections[0].location_data.relative_bounding_box.height,
#         ],
#         [img_w, img_h, img_w, img_h]).astype(int)
        else:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

#             cropped_l_brow = image[mesh_points[LEFT_BROW[0]][1]:mesh_points[LEFT_BROW[1]][1],mesh_points[LEFT_BROW[0]][0]:mesh_points[LEFT_BROW[1]][0]].copy()
#             cropped_l_brow = cv.resize(cropped_l_brow,(48,24))

#             cropped_r_brow = image[mesh_points[RIGHT_BROW[0]][1]:mesh_points[RIGHT_BROW[1]][1],mesh_points[RIGHT_BROW[0]][0]:mesh_points[RIGHT_BROW[1]][0]].copy()
#             cropped_r_brow = cv.resize(cropped_r_brow,(48,24))

            cropped_brow = rgb_frame[mesh_points[BROW[0]][1]:mesh_points[BROW[1]][1],mesh_points[BROW[0]][0]:mesh_points[BROW[1]][0]].copy()
            cropped_brow = cv.resize(cropped_brow,(96,24))

            cropped_nose = rgb_frame[mesh_points[NOSE[0]][1]:mesh_points[NOSE[1]][1],mesh_points[NOSE[0]][0]:mesh_points[NOSE[1]][0]].copy()
            cropped_nose = cv.resize(cropped_nose,(60,26))

            cropped_lips = rgb_frame[mesh_points[LIPS[0]][1]:mesh_points[LIPS[1]][1],mesh_points[LIPS[0]][0]:mesh_points[LIPS[1]][0]].copy()
            cropped_lips = cv.resize(cropped_lips,(78,28))


            if pt_pos<9:
                cv.circle(frame, (mid_pt[pt_pos][0],mid_pt[pt_pos][1]), 30, (0,255,0), -1)
                img_array = 0
                if "_l_" in modelName:
                    img_array = np.expand_dims(cropped_lips, axis=0)
                elif "_b_" in modelName:
                    img_array = np.expand_dims(cropped_brow, axis=0)
                elif "_n_" in modelName:
                    img_array = np.expand_dims(cropped_nose, axis=0)
                else:
                    img_array = np.expand_dims(cropped_lips, axis=0)
                img = img_array.astype(np.float32) / 255.0
                prediction = model.predict(img,verbose=0)
                prediction = prediction[0][0]
                cv.putText(frame, "confidence: "+str(prediction), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                if prediction>0.55:
                    if success_cnt==3:
                        font = ImageFont.truetype(fontpath, 35)      
                        imgPil = Image.fromarray(frame)                
                        draw = ImageDraw.Draw(imgPil)                
                        draw.text((mid_pt[pt_pos][0]+40,mid_pt[pt_pos][1]-30), "偵測到微表情", fill=(0, 0, 255), font=font)
                        frame = np.array(imgPil)
                        success_cnt = 0
                    else:
                        success_cnt+=1
            else:
                font = ImageFont.truetype(fontpath, 35)      
                imgPil = Image.fromarray(frame)                
                draw = ImageDraw.Draw(imgPil)                
                draw.text((int(height/2)-100,int(width/2)-100), "正確率為"+str(correct)+"/9，請按q鍵關閉系統", fill=(0, 255, 0), font=font)
                frame = np.array(imgPil)
            cv.imshow('subtle facial',frame)

            key = cv.waitKey(1) & 0xFF
                # q for exit
            if key == ord('q'):
                break
            elif key == ord('y'):
                pt_pos+=1
                correct+=1
            elif key == ord('n'):
                pt_pos+=1
        
        

cap.release()
cv.destroyAllWindows()

# 少收一點資料看看，接下來跟panel結合multithreading(1/16)，加入continual learning
    