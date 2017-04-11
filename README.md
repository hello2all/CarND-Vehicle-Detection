## Udacity SDCND Project 5
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 

* Normalize feature vector before feed into a linear Support Vector Machine classifier, train and test model performance

* Implement a sliding-window technique and use trained classifier to search for vehicles in images.

* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/ROI.png
[image4]: ./examples/sliding_windows.png
[image5]: ./examples/pipeline1.png
[image6]: ./examples/pipeline2.png
[image7]: ./examples/pipeline3.png
[image8]: ./examples/bboxes_and_heat.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 15th code cell of the demo.ipynb IPython notebook (or in lines 109 through 147 of the file called `lib.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space in Y channel and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the following parameter values:

| HOG parameter   | Value |
|:---------------:|:-----:|
| Color space     | YCrCb |
| Orientation     | 12    |
| pixels per cell | 8     |
| cell per block  | 1     |
| Channels        | All   |

For color features, the following parameters are selected through trial and error

| Color parameter | Value |
|:---------------:|:-----:|
| Color space     | HLS   |
| histogram bins  | 100   |
| histogram range | 0-256 |
| binned size     | 32x32 |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the labeled data from [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).

The labeled data consists of two labels: car and none car. The data set is split into training set and testing set, which accounts 90% and 10% of the dataset respectively.

A linear Support Vector Machine classifier then is implemented to learn from the feature vector to predict image labels. With C parameter set to 1e-4, the final model has a accuracy of 99.71% and F1 score of 0.9970. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions with vertical range from 350 to 656 as the region of interest. The ROI is illustrated in the image below.

![alt text][image3]

Then, 5 windows with scale between 1.25 to 2.5 slides through the region of intrest to detect whether a car exists in the current window. The code for this function is located in `pipe_line.py` line 28-42. An illustration of the sliding windows is shown below.

![alt text][image4]

The scale of the window is tested on mutiple images through trial and error. When sliding the windows, 2 cells of hog block is skipped between every two boxes. Overall the sliding windows have overlap between 60-85%.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap, threshold=1 is applied to filter out some of the false positives.

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems/ issues: 

- Hand crafted features are very easy to break, there are many moving parts that could fail. Lot of time is allocated on adjusting parameters.

- False positives are very difficult to get rid of without sacrifising true possitives.

Future direction:

- I'll latter try to tackle this problem using a different approach, such as [YOLO](https://pjreddie.com/media/files/papers/yolo.pdf) or [SSD](https://arxiv.org/pdf/1512.02325.pdf).

