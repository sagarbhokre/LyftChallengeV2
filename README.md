# LyftChallengeV2 - Semantic Segmentation

### Overview
Implementation tries to address semantic segmentation requirement for Lyft Challenge with Udacity. It uses [MobileNet](https://arxiv.org/abs/1704.04861) architecture. All stride-16 depthwise convolutions were replaced with dilated depthwise convolutions and the two final stride-32 layers were removed. This is similar to what was done in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122). It also contains skip-connections for stride-8 and stride-4 (adding stride-2 gave no better results).
The model uses the [Keras MobileNet implementation](https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py) and training is done with [TensorFlow](https://www.tensorflow.org/).

### Preparing the data for training
mkdir data
cd data
wet https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz -O dataset.tar.gz
tar -xvzf dataset.tar.gz

Train/Validation split: 90 % / 10 %

Original dataset used for training [Not used for the challenge]:

[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) with 90% / 10% training/validation data split. Images are downscaled to half resolution.

[//]: # (Image References)
[image1]: ./res/loss_curves.png
[image2]: ./res/augmentation_methods_overview.png
[image3]: ./res/latest_run.png
[image4]: ./res/benchmark_results.png

#### Data augmentation
The dataset is rather small i.e. only 1000 training images. A few augmentation methods include: rotation, flipping, blurring and changing the illumination of the scene (see `augmentation.py`).
An example is given in the following image:
![alt text][image2]

#### Quantitative Results
Training for 100 epochs results in the following loss curves:
![alt text][image1]
I ran a few experiments with different learning rate schedules, varying amounts of data augmentation and changed dilation rates but the results did not change that much. Moreover, there was no big difference in training from scratch vs. using ImageNet weights.
By default logs are saved to the `log` directory. To visualize them start tensorboard via `tensorboard --logdir log`.

#### Qualitative Results
Here are a few predictions @~92% validation mIoU:
![alt text][image3]
It can be seen that shadows are handled quite well. Road vs. sidewalk still leaves room for improvements (e.g. bottom left). More results can be found in the `latest_run` directory.

<!---
### Inference Optimization
There are plenty of [methods for inference optimization](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) already implemented in tensorflow. I am using the `fold_constants`, `fold_batch_norms` and `round_weights` transforms (see `export_for_mobile.py`).
These methods are explained in more detail on the [Pete Warden blog](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/).
After all, the (zipped) optimized graph is only 1.8M.
To optimize your graph run:
```
python3 export_for_mobile.py - -weight_path ep-000-val_loss-0.0000.hdf5
```
You can check and benchmark the optimized graph with:
```
python3 test_graph.py
` ``
You should see that you get nice (tiny) speed-ups on the CPU (GPU) while the results stay the same:
` ``
One forward pass for `freezed.pb` on `/gpu:0` took: 20.6018 ms
One forward pass for `optimized.pb` on `/gpu:0` took: 20.5921 ms
One forward pass for `freezed.pb` on `/cpu:0` took: 163.9354 ms
One forward pass for `optimized.pb` on `/cpu:0` took: 146.2478 ms
` ``
Though, I only checked them visually:
![alt text][image4]

**Note**: Most of the app code comes from these two codelabs:
- [TensorFlow for Poets 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/index.html?index=..%2F..%2Findex#0)
- [Android & TensorFlow: Artistic Style Transfer](https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html?index=..%2F..%2Findex#0)

#### Quantization
As proposed in the chapter 'Quantization Challenges' of [Building Mobile Applications with TensorFlow](http://www.oreilly.com/data/free/building-mobile-applications-with-tensorflow.csp) I implemented the 8-bit quantization. Unfortunately, one can't just use the python interface. The procedure is as follows:
- Freeze the graph:
` ``
python3 export_for_roady.py - -weight_path ep-000-val_loss-0.0000.hdf5
` ``
- Then run:
` ``
bash quantize.sh
` ``

The second step requires that you built the transform_graph binary (`bazel build tensorflow/tools/graph_transforms:transform_graph`). It will record and freeze the requantization ranges.
I tested both versions (8-bit and 32-bit) and the 32-bit version still runs faster.
-->

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [Keras](https://keras.io/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv.org/)

##### Dataset
[Not used for the challenge] Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training and test images.

Please refer to "Preparing the data for training" for more details.

### Start
Run the following command to start the project:
```
python3 main.py
```

Inspired from https://github.com/see--/P12-Semantic-Segmentation
