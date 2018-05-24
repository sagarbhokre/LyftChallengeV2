import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from keras.utils import get_file
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from IPython import embed
import cv2

from common import *

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [
        vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',  # noqa
                os.path.join(
                    vgg_path,
                    vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def maybe_download_mobilenet_weights(alpha_text='1_0', rows=224):
    base_weight_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'  # noqa
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
    weigh_path = base_weight_path + model_name
    weight_path = get_file(model_name,
                           weigh_path,
                           cache_subdir='models')
    return weight_path

def gen_lyft_batches_functions(data_folder, image_shape, image_folder='image_2', label_folder='gt_image_2',
                               train_augmentation_fn=None,
                               val_augmentation_fn=None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = []
    label_fns = []

    data_folders = glob(data_folder+"/*/")

    for data_folder in data_folders:
        image_paths.extend(sorted(glob(os.path.join(data_folder, image_folder, '*.png')))[:])
        label_fns.extend(glob(os.path.join(data_folder, label_folder, '*.png')))

    #image_paths = image_paths[:16]
    train_paths, val_paths = train_test_split(
        image_paths, test_size=0.1, random_state=21)

    #label_paths = {os.path.basename(path): path for path in label_fns}
    label_paths = {path: path for path in label_fns}

    def get_batches_fn(batch_size, image_paths, augmentation_fn=None):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        background_color = np.array([0, 0, 0])
        pedestrian_id = 4
        road_id = 7
        lane_id = 6
        car_id = 10
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[image_file.replace('CameraRGB', 'CameraSeg')]

                in_image = scipy.misc.imread(image_file, mode='RGB')

                #image = scipy.misc.imresize(in_image, image_shape, interp='nearest')
                image = in_image[-image_shape[0]:, :]

                in_gt = scipy.misc.imread(gt_image_file)

                #gt_image = scipy.misc.imresize(in_gt, image_shape, interp='nearest')[:,:,0]
                gt_image = in_gt[-image_shape[0]:, :, 0]

                gt_road = ((gt_image == road_id) | (gt_image == lane_id))
                gt_car = (gt_image == car_id)
                gt_car[491:,:] = False
                gt_bg = np.invert(gt_car | gt_road)

                if augmentation_fn:
                    image, gt_bg, gt_car, gt_road = augmentation_fn(image, gt_bg, gt_car, gt_road)

                if len(images) == 0:
                    cv2.imwrite("GT_car_img.png", 255*gt_car)
                    cv2.imwrite("GT_road_img.png", 255*gt_road)
                    cv2.imwrite("GT_bg_img.png", 255*gt_bg)

                gt_bg= gt_bg.reshape(*gt_bg.shape, 1)
                gt_car= gt_car.reshape(*gt_car.shape, 1)
                gt_road = gt_road.reshape(*gt_road.shape, 1)

                gt_image = np.concatenate((gt_bg, gt_car, gt_road), axis=2)

                '''
                image_l = scipy.misc.toimage(image)
                arg_label = gt_image.argmax(axis=2)
                img = blend_output(image_l, arg_label, (255,0,0), (0,255,0))
                cv2.imshow("Out", np.array(img))
                c = cv2.waitKey(0) & 0x7F
                if c == 27 or c == ord('q'):
                    exit()
                '''

                images.append(image)
                gt_images.append(gt_image)

            #yield np.array(images) / 127.5 - 1.0, np.array(gt_images)
            yield np.array(images), np.array(gt_images)

    train_batches_fn = lambda batch_size: get_batches_fn(batch_size, train_paths, augmentation_fn=train_augmentation_fn)  # noqa
    val_batches_fn = lambda batch_size: get_batches_fn(batch_size, val_paths, augmentation_fn=val_augmentation_fn)  # noqa

    return train_batches_fn, val_batches_fn, len(train_paths), len(val_paths)



def gen_batches_functions(data_folder, image_shape, image_folder='image_2', label_folder='gt_image_2',
                          train_augmentation_fn=None,
                          val_augmentation_fn=None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = sorted(
        glob(os.path.join(data_folder, image_folder, '*.png')))[:]
    train_paths, val_paths = train_test_split(
        image_paths, test_size=0.1, random_state=21)

    def get_batches_fn(batch_size, image_paths, augmentation_fn=None):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        label_fns = glob(os.path.join(
            data_folder, label_folder, '*_road_*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in label_fns}

        background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(
                    scipy.misc.imread(image_file, mode='RGB'), image_shape, interp='nearest')

                gt_image = scipy.misc.imresize(
                    scipy.misc.imread(gt_image_file), image_shape, interp='nearest')

                gt_bg = np.all(gt_image == background_color, axis=2)
                if augmentation_fn:
                    image, gt_bg = augmentation_fn(image, gt_bg)

                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images) / 127.5 - 1.0, np.array(gt_images)

    train_batches_fn = lambda batch_size: get_batches_fn(batch_size, train_paths, augmentation_fn=train_augmentation_fn)  # noqa
    val_batches_fn = lambda batch_size: get_batches_fn(batch_size, val_paths, augmentation_fn=val_augmentation_fn)  # noqa

    return train_batches_fn, val_batches_fn


def blend_output(frame, im_out, c, r):
    image_shape = frame.size

    #car_segmentation = (im_softmax[:,:,0] > 0.5).reshape(image_shape[0],image_shape[1],1)
    car_segmentation = np.where((im_out==CAR_ID),1,0).astype('uint8').reshape(image_shape[1],image_shape[0],1)
    car_mask = np.dot(car_segmentation, np.array([[c[0], c[1], c[2], 127]]))
    car_mask = scipy.misc.toimage(car_mask, mode="RGBA")

    road_segmentation = np.where((im_out==ROAD_ID),1,0).astype('uint8').reshape(image_shape[1],image_shape[0],1)
    road_mask = np.dot(road_segmentation, np.array([[r[0], r[1], r[2], 127]]))
    road_mask = scipy.misc.toimage(road_mask, mode="RGBA")

    #ped_segmentation = np.where((im_out==2),1,0).astype('uint8').reshape(image_shape[0],image_shape[1],1)
    #ped_mask = np.dot(ped_segmentation, np.array([[0, 0, 255, 77]]))
    #ped_mask = scipy.misc.toimage(ped_mask, mode="RGBA")

    street_im = scipy.misc.toimage(frame)
    street_im.paste(road_mask, box=None, mask=road_mask)
    street_im.paste(car_mask, box=None, mask=car_mask)
    #street_im.paste(ped_mask, box=None, mask=ped_mask)

    return street_im


def get_seg_img(sess, logits, image_pl, pimg_in, image_shape, learning_phase):

    new_shape = [x // 16 * 16 for x in image_shape]
    d = image_shape[0] - new_shape[0]

    pimg = pimg_in[d:,:,:]
    im_softmax = sess.run(tf.nn.softmax(logits), {image_pl: [pimg], learning_phase: 0})

    im_softmax = im_softmax.reshape(new_shape[0], new_shape[1], -1)
    im_out = im_softmax.argmax(axis=2)

    image = (pimg_in + 1.0) * 127.5
    image = scipy.misc.toimage(image)

    return blend_output(image, im_out, (0,255,0), (0,0,0))

def gen_lyft_test_output(
        sess,
        logits,
        image_folder,
        image_pl,
        data_folder,
        learning_phase,
        image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    for image_file in sorted(
            glob(os.path.join(data_folder, image_folder, '*.png')))[:]:

        in_image = scipy.misc.imread(image_file, mode='RGB')
        image = scipy.misc.imresize(in_image, image_shape)

        pimg = image / 127.5 - 1.0

        street_im = get_seg_img(sess, logits, image_pl, pimg, image_shape, learning_phase)

        street_im = scipy.misc.imresize(street_im, in_image.shape)
        yield os.path.basename(image_file), np.array(street_im)

def gen_test_output(
        sess,
        logits,
        image_pl,
        data_folder,
        learning_phase,
        image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in sorted(
            glob(os.path.join(data_folder, 'image_2', '*.png')))[:]:
        image = scipy.misc.imresize(
            scipy.misc.imread(image_file, mode='RGB'), image_shape)
        pimg = image / 127.5 - 1.0
        im_softmax = sess.run(
            tf.nn.softmax(logits),
            {image_pl: [pimg],
             learning_phase: 0})
        im_softmax = im_softmax[:, 1].reshape(
            image_shape[0], image_shape[1])
        segmentation = (
            im_softmax > 0.5).reshape(
            image_shape[0],
            image_shape[1],
            1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(
        runs_dir,
        data_dir,
        sess,
        image_shape,
        logits,
        learning_phase,
        input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, input_image, os.path.join(
            data_dir, 'data_road/testing'), learning_phase, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def lyft_save_inference_samples(
        runs_dir,
        data_dir,
        image_folder,
        sess,
        image_shape,
        logits,
        learning_phase,
        input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_lyft_test_output(
        sess, logits, image_folder, input_image, os.path.join(
            data_dir, 'Test'), learning_phase, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
