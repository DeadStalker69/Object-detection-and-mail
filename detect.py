import cv2
import time

from RPLCD.i2c import CharLCD

# lcd = CharLCD(i2c_expander = 'PCF8574', address = 0x27, port = 1, cols = 16, rows = 2, dotsize = 8)
# lcd.clear()

import RPi.GPIO as gpio

gpio.setwarnings(False)
gpio.setmode(gpio.BCM)
gpio.setup(14, gpio.OUT)
gpio.output(14, False)

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

fromaddr = 'prabhatg3356@gmail.com'
toaddr = 'prabhatg3356@gmail.com'

msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Booth Violation"
body = 'More than one person was found at the EVM voting booth. Attached is the image from the booth.'
msg.attach(MIMEText(body, 'plain'))

filename = "Image.jpg"

part = MIMEBase('application', 'octet-stream')

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "lmyr sytc hpcv mpvd")
text = msg.as_string()

model='efficientdet_lite0.tflite'
num_threads=4

dispW=1280
dispH=720

webCam='/dev/video0'
cam=cv2.VideoCapture(webCam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)

pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)

label_height = 1.5
label_color = (0,255,0)
label_weight = 2

box_color = (0,0,255)
box_weight = 2

fps=0

base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
detection_options=processor.DetectionOptions(max_results=3, score_threshold=.5)
options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)
tStart=time.time()
while True:
    ret, im = cam.read()
    imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imTensor=vision.TensorImage.create_from_array(imRGB)
    myDetects=detector.detect(imTensor)
    
    count = 0
    for myDetect in myDetects.detections:
        UL = (myDetect.bounding_box.origin_x, myDetect.bounding_box.origin_y)
        LR = (myDetect.bounding_box.origin_x + myDetect.bounding_box.width, myDetect.bounding_box.origin_y + myDetect.bounding_box.height)
        objName = myDetect.categories[0].category_name
        im = cv2.rectangle(im, UL, LR, box_color, box_weight)
        cv2.putText(im, objName, UL, font, label_height, label_color, label_weight)
        
        if(objName == 'person'):
            count = count + 1
#             lcd.write_string('Person\r\n')
#             lcd.write_string('Detected')
#             
        #else:
#             lcd.clear()
#             
#         if(objName == 'cell phone'):
#             gpio.output(14, True)
#             
#         else:
#             gpio.output(14, False)
    if(count > 1):
        gpio.output(14, True)
        cv2.imwrite("Image.jpg", im)
        
        attachment = open("Image.jpg", 'rb')
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename)
        msg.attach(attachment_package)
        
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        
    else:
        gpio.output(14, False)
    #image=utils.visualize(im, myDetects)
    cv2.putText(im,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    cv2.imshow('Camera',im)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps= .9*fps +.1*1/loopTime
    tStart=time.time()
cv2.destroyAllWindows()
