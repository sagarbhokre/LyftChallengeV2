import cv2
import utils
import sys
import numpy as np
import os
path = os.getcwd()
sys.path.append(path+'/..')
from common import *

visualize = True
create_video = True
videoout = None

def plot_data(flist):
    if create_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        videoout = cv2.VideoWriter('visualization_output.avi',fourcc, 30.0, (input_width, input_height))

    idx = 0
    while True:
        if idx >= len(flist):
            print("Done", flist[0].split('/')[-3])
            exit()
        img = cv2.imread(flist[idx])
        fname_seg = flist[idx].replace('CameraRGB', 'CameraSeg')
        seg = cv2.imread(fname_seg)[:,:,2]

        out = np.zeros_like(img)*150

        out[:,:,1] = np.where(seg == 10, 255, 0).astype('uint8')
        out[HOOD_OFFSET:,:,1] = 0

        out[:,:,2] = np.where(((seg == 7) | (seg == 6)), 251, 0).astype('uint8')

        overshoot = np.sum(out[:OFFSET_HIGH,:,1])
        if overshoot > 0:
            print(flist[idx])

        if not create_video:
            out[:OFFSET_HIGH,:,:] = 0
            out[OFFSET_LOW:,:,:] = 0

        if visualize:
            alpha = 0.4
            cv2.addWeighted(out, alpha, img, 1 - alpha, 0, out)

            cv2.putText(out, flist[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Overlay", out)
            cv2.moveWindow("Overlay", 20,20);

            if videoout is not None:
                videoout.write(out)

            c = cv2.waitKey(0) & 0x7F
            if c == 27:
                exit()
            elif c == 83:
                idx += 1
            elif c == 81:
                idx -= 1
            else:
                idx += 1
            idx %= len(flist)
        else:
            idx += 1

    if create_video:
        videoout.release()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s <img_folder>"%(sys.argv[0]))
        exit()

    flist = utils.get_files(sys.argv[1]+'/CameraRGB/')
    plot_data(flist)
