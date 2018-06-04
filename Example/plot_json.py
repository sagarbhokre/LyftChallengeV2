#!/bin/python

import sys
import json
import base64
import numpy as np
from scipy import misc
import cv2

student_output = sys.argv[1]

def decode(packet):
    img = base64.b64decode(packet)
    filename = 'image.png'
    with open(filename, 'wb') as f:
        f.write(img)
    result = misc.imread(filename)
    return result

# Load student data
with open(student_output) as student_data:
    student_ans_data = json.loads(student_data.read())
    student_data.close()

frames_processed = 0

for frame in range(1,len(student_ans_data.keys())+1):

    student_data_car = decode(student_ans_data[str(frame)][0])
    student_data_road = decode(student_ans_data[str(frame)][1])

    cv2.imshow("Car", student_data_car*250)
    cv2.imshow("Road", student_data_road*250)
    c = cv2.waitKey(0) & 0x7F
    if c == 27 or c == ord('q'):
        exit()

    frames_processed+=1

