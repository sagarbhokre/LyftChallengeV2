import cv2
import utils
import sys
import numpy as np
import os
path = os.getcwd()
sys.path.append(path+'/..')
from common import *
import json
import base64
import numpy as np
from scipy import misc

visualize = True
create_video = True
videoout = None

def decode(packet):
    img = base64.b64decode(packet)
    filename = '/home/workspace/Example/image.png'
    with open(filename, 'wb') as f:
        f.write(img)
    result = misc.imread(filename)
    return result


def plot_data(video_in, json_in):
    global videoout
    if create_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        videoout = cv2.VideoWriter('visualization_output.avi',fourcc, 10.0, (input_width, input_height))

    with open(json_in) as json_data:
        ans_data = json.loads(json_data.read())
        json_data.close()

    cap = cv2.VideoCapture(video_in)

    idx = 1
    skip_first_frame = False
    read_count = 0
    dump_enabled = True
    print("dump status: ", dump_enabled)
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        read_count += 1

        if skip_first_frame:
            skip_first_frame = False
            continue

        seg = np.zeros_like(img)

        truth_data_car =  decode(ans_data[str(idx)][0])
        truth_data_road =  decode(ans_data[str(idx)][1])

        seg[:,:,1] = truth_data_car*250
        seg[:,:,2] = truth_data_road*250
        seg[HOOD_OFFSET:,:,1] = 0

        if visualize:
            alpha = 0.4
            cv2.addWeighted(seg, alpha, img, 1 - alpha, 0, seg)
            out = seg

            cv2.putText(out, str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Overlay", out)
            cv2.moveWindow("Overlay", 20,20);

            if dump_enabled and videoout is not None:
                videoout.write(out)

            c = cv2.waitKey(1) & 0x7F
            if c == 27:
                exit()
            elif c == ord('t'):
                dump_enabled = not dump_enabled
                print("dump status: ", dump_enabled)
            idx += 1
        else:
            idx += 1

    # When everything done, release
    if create_video:
        videoout.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: %s <video_file> <json_file>"%(sys.argv[0]))
        exit()

    video_in = sys.argv[1]
    json_in = sys.argv[2]
    plot_data(video_in, json_in)
