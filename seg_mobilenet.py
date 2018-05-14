from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers.merge import Add
from keras.applications.mobilenet import DepthwiseConv2D
from keras.applications.mobilenet import relu6
from keras.applications.mobilenet import _conv_block

import tensorflow as tf


def TruncatedMobileNet(input_height, input_width):
    assert input_height // 16 * 16 == input_height
    assert input_width // 16 * 16 == input_width
    alpha = 1.0
    depth_multiplier = 1
    img_input = Input(shape=[input_height, input_width, 3], name='image_input')
    # s / 2
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x_s2 = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    # s / 4
    x = _depthwise_conv_block(x_s2, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x_s4 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    # s /  8
    x = _depthwise_conv_block(x_s4, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x_s8 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    # s / 16
    # DONE(see--): Make some of these dilated
    x = _depthwise_conv_block(x_s8, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, dilation_rate=1, block_id=7)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, dilation_rate=1, block_id=8)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, dilation_rate=2, block_id=9)
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, dilation_rate=4, block_id=10)
    x_s16 = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, dilation_rate=4, block_id=11)

    return img_input, x_s2, x_s4, x_s8, x_s16


def SegMobileNet(input_height, input_width, num_classes=21):
    assert input_height // 16 * 16 == input_height
    assert input_width // 16 * 16 == input_width
    img_input, x_s2, x_s4, x_s8, x_s16 = TruncatedMobileNet(
        input_height, input_width)
    x_s16 = _conv_bn_pred(x_s16, 16, num_classes=num_classes)
    Upsample_s8 = Lambda(
        lambda x: _resize_bilinear(x, input_height // 8, input_width // 8))
    x_up8 = Upsample_s8(x_s16)
    x_s8 = _conv_bn_pred(x_s8, 8, num_classes=num_classes)
    x_s8 = Add(name='add_s8')([x_up8, x_s8])
    Upsample_s4 = Lambda(
        lambda x: _resize_bilinear(x, input_height // 4, input_width // 4))
    x_up4 = Upsample_s4(x_s8)
    x_s4 = _conv_bn_pred(x_s4, 4, num_classes=num_classes)
    x_s4 = Add(name='add_s4')([x_up4, x_s4])
    Upsample_s2 = Lambda(
        lambda x: _resize_bilinear(x, input_height // 2, input_width // 2))
    x_up2 = Upsample_s2(x_s4)
    # x_s2 = _conv_bn_pred(x_s2, 2, num_classes=num_classes)
    # x_s2 = Add(name='add_s2')([x_up2, x_s2])
    x_s2 = x_up2
    Upsample_s1 = Lambda(
        lambda x: _resize_bilinear(x, input_height // 1, input_width // 1))
    x = Upsample_s1(x_s2)
    return Model(img_input, x, name='SegMobileNet')


def _conv_bn_pred(x, stride, num_classes=21):
    x = Conv2D(num_classes, 1, use_bias=False, padding='same',
               name='conv_pred_s%d' % stride)(x)
    x = BatchNormalization(name='conv_pred_s%d_bn' % stride)(x)
    return x


def _resize_bilinear(x, target_h, target_w):
    x = tf.image.resize_images(
        x,
        [target_h, target_w],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)
    x.set_shape((None, target_h, target_w, None))
    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1,
                          dilation_rate=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    DONE(see--): Allow dilated depthwise convolutions
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id,
                        dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
