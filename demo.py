import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import cv2
import scipy.misc
import time

from seg_mobilenet import SegMobileNet
from keras.models import load_model
import numpy as np
import helper

n_classes = 3
input_width = 800
input_height = 600
visualize = False
enable_profiling = False
image_shape = (input_height, input_width)

model_path = 'checkpoint/ep-029-val_loss-0.5531.hdf5'

def load_seg_model():
    new_shape = [x // 16 * 16 for x in image_shape]
    m = SegMobileNet(input_height=new_shape[0], input_width=new_shape[1], num_classes=n_classes)
    m.load_weights(model_path)
    m.compile(loss='categorical_crossentropy',
              optimizer= 'adadelta' ,
              metrics=['accuracy'])

    nw_output_shape = m.output[0].shape

    output_height = nw_output_shape[0]
    output_width = nw_output_shape[1]

    return m, output_width, output_height

save_count = 0
def visualizeImage(rgb_frame, im_out, render=True):
    street_img = helper.blend_output(rgb_frame, im_out)

    if render:
        global save_count
        #scipy.misc.imshow(street_im)
        scipy.misc.imsave('dump/'+str(save_count)+'.png', street_im)
        save_count += 1
        #c = cv2.waitKey(30) & 0x7F
        #if c == ord('q') or c == 27:
            #exit() 

def preprocess_img(img, ordering='channels_first'):
    #in_image = scipy.misc.imread(image_file, mode='RGB')
    #img = scipy.misc.imresize(img, image_shape)
    img = img / 127.5 - 1.0
    return img

if __name__ == '__main__':
    file = sys.argv[-1]

    if file == 'demo.py':
      print ("Error loading video")
      quit

    # Define encoder function
    def encode(array):
        pil_img = Image.fromarray(array)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    video = skvideo.io.vread(file)

    m, output_width, output_height = load_seg_model()

    answer_key = {}

    # Frame numbering starts at 1
    frame = 1

    start_t = time.time()

    pr = np.ones((input_height, input_width))*2

    for rgb_frame in video:
        d = int(rgb_frame.shape[0] - int(output_height))

        frame_shape = rgb_frame.shape
        X = preprocess_img(rgb_frame[d:,:,:])

        pr_out = m.predict( np.array([X]) )[0]

        pr[d:,:] = pr_out.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        #pr = pr_out.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        #pr = np.pad(pr, ((d,0), (0,0)), 'edge')
        binary_car_result  = np.where((pr==0),1,0).astype('uint8')
        binary_road_result = np.where((pr==1),1,0).astype('uint8')

        #binary_car_result  = np.where((pr_out[:,:,0]>0.5),1,0).astype('uint8')
        #binary_road_result = np.where((pr_out[:,:,1]>0.5),1,0).astype('uint8')
        answer_key[frame]  = [encode(binary_car_result), encode(binary_road_result)]

        if visualize:
            seg_img = visualizeImage(rgb_frame, pr, render=True)

        # Increment frame
        frame+=1

    if enable_profiling:
        fps = frame/(time.time() - start_t)
        print("FPS: %f"%(fps))
    else:
        # Print output in proper json format
        print (json.dumps(answer_key))
