# LyftChallengeV2 - Semantic Segmentation

### Introduction to Segmentation and Lyft Challenge
Image Segmentation is the process of labelling each pixel in the image as belonging to a predefined class. In this project, the task is to label each pixel in the image as Car, Road or Background.

Various methods can be employed to get the segmentation task done but the Challenge demanded a unique set of performance metrics. One metric was F_score and the other was execution speed (frames per second – fps)

The implementation detailed below uses a Deep Neural Network (DNN) to attain the segmentation goal. Reason for selecting a DNN over vision approach was simple, there is no need for manually engineering the features and DNNs can exploit parallelism way better than vision approaches. With the execution speed being a metric DNN appeared to be a sensible choice

Using the DNN solution requires 
1.	Deciding on a DNN architecture
2.	Defining the input and output parameters
3.	Collecting data for training the network
4.	Writing code for training, testing and visualizing the parameters of the neural network
5.	Training the neural network
6.	Fine tuning to optimize the accuracy (F_score for this challenge)
7.	Fine tuning to improve speed
8.	Local infrastructure to evaluate the score


#### Deciding on a DNN architecture:
Multiple choices of DNNs are available these days with architectures ranging from fast and lightweight to slow and bulky depending on the computation and accuracy requirements. The network architectures I considered included UNet, MobileNet, VGGNet, GoogleNet. Out of these choices, MobileNet and UNet appeared promising. [MobileNet](https://arxiv.org/abs/1704.04861) architecture of the two, was lightweight and gave good performance as per the latest reported values on ImageNet. It also allowed finer control on the number of layers, size of layers and was very light weight. I proceeded with MobileNet architecture with the following changes; Fully connected layers were removed from the final DNN, Conv2DTranspose layers with a stride of (2,2) were added to the tail of the truncated MobileNet and output size was brought back to input dimensions. Upsampling was possible using two options bilinear interpolation or Conv2DTranspose; bilinear was a crude method and offered no scope for learning/fine-tuning, Conv2DTranspose on the other hand would learn from the input data and fine-tune its performance depending on the input at a small computation cost. Conv2DTranspose was the choice I made. Skip connections were added at scale factors of 2,4,8 and 16 to retain information and improve accuracy. These connections also partially address the vanishing gradient problem allowing faster learning and finer predictions. All stride-16 depthwise convolutions were replaced with dilated depthwise convolutions and the two final stride-32 layers were removed. This is similar to what was done in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122). It also contains skip-connections for stride-8 and stride-4. This implementation is highly inspired from https://github.com/see--/P12-Semantic-Segmentation

The model uses the [Keras MobileNet implementation](https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py) and training is done with [TensorFlow](https://www.tensorflow.org/).

Cross entropy loss (ce_loss) was used as a starting point. Later, after analyzing the DNN performance on training data, I added F_score loss as well. The rationale for adding F_score is partially explained in the Data Visualization section.

### Preparing the data for training
1000 images were provided as a starting point to train the DNN of choice. Please refer to the instructions mentioned below for accessing the dataset

mkdir data
cd data
wet https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz -O dataset.tar.gz
tar -xvzf dataset.tar.gz

After playing around with these 1000 images I realized the data was not enough to generalize the solution. Following section details collecting more data using the CARLA simulator

### Using CARLA simulator to capture more data
DNNs work better with more data. The dataset provided along with this challenge contained a total of 1000 images for 5 different scenarios in Town01. This data was good to start with, but the accuracy of the solution saturates and does not improve. Hence, for a generalized solution that could work better in unknown scenarios, data collection was a no-brainer. CARLA simulator details are as mentioned here. [CARLA SIMULATOR](http://carla.readthedocs.io/en/latest/getting_started/). Since data capture with the PythonClient worked on a scenario basis (Client script collects data for a given scenario and exits), I modified the client application to collect data for all weather scenarios at different locations and ran the client application to collect all data at once. 

The data collection script generates a total of 100 images/episode * 5 episodes/weather_condition * 15 weather_conditions = 7500 images in one go.

Two such executions were used generating a total of 15000 images for training the DNN. This was one of the keys for improving the generalization capability of the network and hence the score on the Leaderboard.

One more execution of similar type mentioned above was used to generate a validation set. 

The validation set had around 3750 images( 50 images/episode * 5 episodes/weather_condition * 15 weather_conditions)

Command for server:
```ruby
./CarlaUE4.sh  /Game/Maps/Town01 -windowed -ResX=800 -ResY=600 -carla-server
./CarlaUE4.sh  /Game/Maps/Town02 -windowed -ResX=800 -ResY=600 -carla-server
```

Command for python client:
```ruby
python PythonClient/client_example.py -a -i -c Custom.CarlaSettings.ini
```

### Data augmentation
To make the DNN robust and capable enough to handle unknown simulator data during testing, following data augmentation techniques were used.
1.	Horizontal flip: Both input RGB image and the Segmentation image were horizontally flipped with a 50 % probability. This did not lead to addition in the number of images and helped add variation to the existing data
2.	Rotation: while the car turns at a junction, I could see a tilt in the entire car chassis. If the car in front can tilt, so can the car we are driving. With this consideration, rotation of the input and segmentation image together by a max range of 15 degrees was added as an augmentation method.
3.	Hue variation: Only CameraRGB images were subjected to this augmentation technique. It was observed that in different weather situations the color of same objects varied depending on position of the Sun. Say a blue car was brighter during noon and was darker in the evening. This augmentation technique would allow the network to learn variations in color even if that weather scenario was not present in the dataset
4.	Brightness variation: Again, only CameraRGB images were subjected to this augmentation technique. Segmentation images were not altered. When the sun is high up in the sky the images are crisp and bright, but as we Sun lowers near the horizon, it shines directly on the camera saturating its pixels and increasing the overall brightness of the entire image. Intuitively, this augmentation technique made sense as brightness variations would be present in image data.
5.	Contrast variation: Same rationale as the one used for brightness augmentation.
6.	Grayscale conversion: This technique is not used in the implementation; however, the reasoning behind considering this augmentation technique was making the network agnostic to color variations in the image. This augmentation technique would allow the network to learn variations in the image rather than the color. For e.g: the network would try to learn what a car’s shape looks like and what roads shape look like rather than learning what should be a car’s color and what should be the road’s color. It would only make the network more generic and capable of working in a distinct color domain. On implementation, I did not see much improvement in DNNs accuracy. Hence to avoid computation cost and address only the problems related to the challenge, this augmentation technique was not added
7.	Noise addition: Addition of random noise to the input image would again make the network robust in case there is noise in camera sensor. This being a simulated environment, I expect no noise or very less noise added by the camera sensor. Hence this augmentation method was not given much weightage while training the network given that the network would be evaluated in a simulated environment. 

[//]: # (Image References)
[image1]: ./res/loss_graph.png
[image2]: ./res/augmentation.png
[image3]: ./res/latest_run.png
[image4]: ./res/benchmark_results.png
[image5]: ./res/model.png

An example is given in the following image:
![alt text][image2]

#### Data visualization
This block proved to be the most interesting and useful block in the implementation. If I learned major mistakes while implementing the solution, I attribute all the credit to this block. Detailing how this block helped me:
1. It helped me add augmentation methods. At the start when I started with data collection, this block helped me understand that even with brightness saturation (very bright image with Sun at the horizon) the car annotations were marked and could potentially be seen by the network. This helped me understand that brightness variations are important in the network needs to work for scenarios where Sun may be at the horizon. 
2. Visualization helped me check what is fed to the network while training and what is generated by the network as output. Overlaying all the 3 images (CameraRGB image, Segmentation ground truth and Segmentation output from the DNN) on top of one another helped me visually understand how things are progressing. It helped me debug an issue with augmentation method as well. When an image is rotated, the newly created blank regions were filled with 0 (0 was the code for cars in the segmentation image at that time). This was creating problems as all blank regions were being learnt as cars by the network. Had it not been for visualization, it would have taken me ages to understand what the problem was.
3. It helped me understand what part of the network output is overlapping with ground truth and where is it that the network needs more attention. While visualizing the output I noticed that the car segmentation was thinner and road segmentation was comparatively thicker. This was happening probably because the network understood, segmenting roads properly was a lower hanging fruit and led to smaller loss quickly. This is a classic example of network bias where the network becomes biased towards abundantly available data. It helped me understand there was a need for some sort of class weightage. It was then I pursued class based loss weightage and F_score loss computation along with cross_entropy loss as the combined loss for training the DNN.
4. There was one more issue I faced while iteratively optimizing the solution. The F_score computed locally would never tally with the leaderboard score. Leaderboard score was always lower (F_score = 90.9) than my local F_score (95.6). Such a situation usually arises when a solution overfits local data and does not generalize well to work in unseen scenarios. I tried various regularization experiments, augmentation techniques, network alterations, addition of dropouts, loss function changes, class weighing, image domain change experiments but all in vein. I resorted to data visualization again to understand what is going wrong and what needs to be done to match local evaluations with leaderboard evaluations. Started with test_video provided in the workspace, downloaded the video and the accompanying .json ground truth data. I wrote a script to overlay the ground truth data on video frames and visualized the result. There! The problem surfaced right in front of me! The video frame contents did not match the ground truth annotations for around 1/3rd the number of frames. There seemed to be some sort of encoding optimization in the video file while led to this situation. That explained why the score I evaluated locally is lower than the one in the leaderboard. On a second thought, the scenario was the same for all contestants and they were all evaluating their solution on the same video. It appeared fair that all contestants were facing the same issue and still managed to score well! Applause!!! This was a good learning with good cost (cost of wasting time debugging the difference in my local evaluation and losing out on the leaderboard score ☹)
5. Another important set of parameters to be visualized included validation metrics; F_car, F_road, F_road. All these parameters were monitored using tensorboard. I added F_car, F_road, F_score and ce_loss to the evaluation metrics. These metrics were evaluated per batch and also added to the tensorboard. You can see in the image below F_road (fr) reaches value 1.0 faster compared to F_car (fc) which converges slowly.

![F_car and F_road visualization example][image1]

6. Input to the network, segmentation output from the network and segmentation ground truth were visualized using tf.summary.image().RGB image formed the base on top of which the input segmentation image and network output segmentation image were overlayed each with an alpha value of 0.4

To visualize tensorboard parameters and images start tensorboard via `tensorboard --logdir log`.

<div align="center">
  <a href="https://www.youtube.com/watch?v=3OZM7wVEam8"><img src="https://img.youtube.com/vi/3OZM7wVEam8/0.jpg" alt="Discrepancy Visualization"></a>
</div>

<!-- [![Discrepancy Visualization](https://img.youtube.com/vi/3OZM7wVEam8/0.jpg)](https://www.youtube.com/watch?v=3OZM7wVEam8 -->

Use the following script to visualize the data. The data is expected to be in following folders
<\_data_folder\_>
 * CameraRGB
   * 1.png
   * 2.png
   * .
   * .
 
 * CameraSeg
   *  1.png
   *  2.png
   *  .
   *  .

```ruby
python3 visualize_data.py <_data_folder_>
```
Similarly video file along with corresponding json ground truth can be visualized as follows:

```ruby
python3 visualize_combined.py <_video_file_> <_json_file_>
```

### Model Architecture in layman terms
As briefed earlier, the solution uses a MobileNet DNN architecture with 4 skip connections. If the network can be split into 2 sections, the encoder and the decoder block, encoder block would be the one which takes in input image and passes it through a series of convolution layers to decipher where the object of interest could be. Decoder block maps this encoded data back to the output dimensions and highlights the object of interest. Encoder block needs to be comparatively complex as it happens to be the main intelligence whereas the decoder block can be lightweight as it needs to just expand the data to output dimensions. Encoder block in the implementation is a Truncated MobileNet with 12 convolution layers and the decoder block consists of 4 Conv2DTranspose layers with each layer creating an output with a scaling factor of 2. Output from each Conv2DTranspose layer is given to an Adder which combines outputs at different scales and generates one plane of image per class. For this implementation, background occupies plane 0, car occupies plane 1 and road data is present in plane 2.


### Loss computation
Cross entropy loss seemed to be an appropriate choice at the start. This allowed the solution to converge and taught the DNN how to segment roads and cars. However, on deeper analysis the loss function did not live up to the expectations as it would make the inferences more biased towards the abundantly available class (road in this case). I experimented with weighted loss function to see if that gave any advantage, it did better than vanilla cross entropy loss but still did not outperform it. Given the challenge and evaluation metric (F_score) , I thought why not use the same metric as a loss function! 

It would serve 2 benefits:
1. I would see in practice how well would my solution do on the leaderboard 
2. The solution would give higher importance to cars compared to roads

Added this loss function to the implementation, and I did see some problems. It did not converge at all to start with. Maybe this had to do with how the loss was computed as the loss computation mathematics would decide backpropagation efficiency. Me as a programmer working to find a quick-dirty solution might not have done it efficiently leading to less efficient backpropagation. As a quick-dirty solution to this problem, I had an optimized reference in place, ‘the cross-entropy loss function’. I combined the two loss functions ce_loss + F_score and used this as a loss function. By combining the two, I could converge fast to start with (ce_loss did well here to segment road data) and as the DNN learns to segment roads properly, ce_loss magnitude would be lower, this is where F_score loss would chip in and optimize for car segmentation at higher priority compared to road segmentation. Phew!

### Optimizations for speed, accuracy:
The final score on leaderboard penalizes slower executions. 
```ruby
fps_penalty = (10 – fps) if fps < 10 
            = 0 if fps > 10
Final score = F_score * 100 – fps_penalty
```
Early in the competition, my fps score was 7.5 with MobileNet architecture defined above. I did a series of experiments to profile the demo code and understand where the bottlenecks are. 

Following were the CPU hoggers in my case:
1.	Image preprocessing: I was scaling the image to -1.0 to 1.0 in the demo code
2.	Network execution time: Network was taking around 50 ms to evaluate per image
3.	Post processing: Output from network was operated to threshold and extract image planes. It was further cleaned to remove car hood annotations

All the three components could be potentially optimized to improve speed
1.	I tried to move the preprocessing to training part and got rid of the scaling requirement. I used the images as they are (without pixel level scaling to range -1.0 to 1.0) and trained the network. The experiment seemed to work good without compromising much on the accuracy. This change reduced execution time by around 3 ms.
2.	Network execution time depends on 2 factors: network size and computation load. Larger the DNN more the chances it would be split and executed leading to larger execution time and more the number of convolution layers, larger would be the computation load as each convolution kernel would need to the entire input layer to generate a new output. The MobileNet architecture is a light weight architecture hence network size optimizations would not help much here. That leaves us with reducing computation load; it can be alleviated by reducing the input size. On analyzing the area spanned by cars and roads in the image, it was clear top and bottom parts of the image can be clipped before feeding them to the DNN. Except for the large Coca-cola trucks, all the images appeared around the center patch of the image. It was only with the consideration for these trucks that I set the upper threshold to 94 pixels from the top. Out of the entire image lower 104 pixels were occupied by the car hood. However, the shape of the hood was not rectangular, it bulged towards the center and was lower towards the extreme left and right. To account for the possibility that road would still be visible in the non-bulged regions, the lower threshold was set to 526 pixels from the top. The network input size was thus reduced from (600, 800, 3) to (432,800). This gave a boost in performance of around 10 ms.
3.	Post processing optimization: Rather than creating a new image every time, I created an image initialized with zeros (background plane) and updated patches of the image with the inference output. For car plane, only (0,96) to (496.800) ROI was updated. For road plane, (0,96) to (526,800) ROI was updated. This gave another performance boost of 2 ms.
With all these changes mentioned above I could execute the solution at around 10.5 fps. This was good enough for the time being. I would have dug deeper had there been a reward for even faster solution. Wasn’t needed, didn’t try!

### Evaluation on validation set:
One more thing I would like to emphasize here is the need for a good validation set. In many of my previous experiments, I used to split the input data into test and train sets using a ratio 80:20 or 90:10. That according to me is mistake, if not a mistake, a bad way of splitting the data if you can generate it. An ideal approach would be defining/collecting a validation set which is meant for validation purposes only. Splitting the training data into two parts has 2 major problems.
1.	Contents of validation set are not precisely known. This would block us from analyzing where is the solution lacking and what needs to be done to improve the performance, accuracy
2.	You would never know if the solution is overfitting. Validation results would always say you are doing great though your solution may not work that well in test conditions

### Learnings:
This challenge has been a fun experience and I learned a lot trying to implement and debug the solution. Created a medium post long ago highlighting all my learnings. Summary of the post:
1.	Understand the problem statement before proceeding
2.	With limited time in hand, make informed decisions on the architecture, augmentation, experiments and optimizations
3.	Visualize as much data as you can. It could be input, learning parameters, output, ground truth, evaluation data etc.
4.	More data the better! The data needs to be diverse!
5.	Tailoring the solution to the needs is not bad. Analyze the input, output and solution to check what extremes are possible. 
https://medium.com/@sagarbhokre/lyft-challenge-255d2d771d51



#### Qualitative Results
Here is the DNN prediction on a set of images. Network reaches mIOU score of 99.4% while training.

<div align="center">
  <a href="https://www.youtube.com/watch?v=BGvMeCzTsIE"><img src="https://img.youtube.com/vi/BGvMeCzTsIE/0.jpg" alt="Output Visualization"></a>
</div>


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
Please refer to "Preparing the data for training" for more details.

### Start
Run the following command to start the project:
```
python3 main.py
```

