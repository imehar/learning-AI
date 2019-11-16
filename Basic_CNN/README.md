### [Score](1st_DNN.ipynb) 0.991
### Convolution

In mathematics, convolution means a function derived from two given functions by integration which expresses how the shape of one is modified by other.

In machine learning, it simple means multiplication of two matrices, one of which is our input image and another is a kernel. The output of this conculation would be another matrix which smaller dimension, the reduction in dimension depends on the size of the kernel used.


### Filters/ Kernels

This is the second matrix used in Convolution along with the input, the filters are generally 3 X 3 matrix, so these would have the receptive field of 9 pixels. The network during training phase finds the values for the filters (which are initialized randomly) so that features are extracted and the network can produce desired output.
The filers are responsible for extracting features from the input, such as edges and gradient, texture, pattern, so on. 

### Epochs 

The term epoch in machine learning refers to number of times the learning model has seen the dataset, i.e. the dataset has been passed to the model for training.


### 1 X 1 Convolution

As the Convolution layers increase the number of channels for each output from the layer increases, to extract and accomodate the increasing number of features. Thus resulting in large numbers channels. Further increase in these channels would be too computationally expensive. Therefore, 1 X 1 Convolution matrix are used to reduce the z-dimension of the input. This would essentially combine the extracted features from all the channels and give that as output.


### 3 X 3 Convolution

A 3 X 3 Convolution is a 3 X 3 matrix as a filter matrix in the Convolution layer. The 3 X 3 filter matrix performs matrix dot product with 3 X 3 dimension of the subsection of the image and give a single value. This process is carried out for each of 3 X 3 subsections of the input, this operations reduces the dimension of input from each side, giving an output of (n-2) X (n-2) , where 'n' is the size of input image


### Feature Maps

When a filter / kernel extracts the features from the input, its output from the dataset contains all the possible features from the data of a single type, consider a filter extracting vertical line feature, its feature map would have all the verical lines from the dataset of different sizes, lengths, colors.


### Activation Function

The ouput of a layer from any model has values in continous float format. The activation function are responsible for deciding which values should be propagated or should be suppresed. It works as thershold.


### Receptive Feild

This is associated with the kernel, it means the number of pixels from which the kernel gets its value. Consider a 3 X 3 filter, at any given time it performs operation with 3 X 3 portion of input, therefore the kernel gets its value from 9 input pixels. So, the receptive field of 3 X 3 filter is 9.
For 5 X 5, the receptive field is 25, this can also be achieved by stacking two 3 X 3 filters. Consider a 5 X 5 image, first 3 X 3 filter will give an output of 3 X 3, second 3 X 3 on this will give 1 X 1, which is similar to one 5 X 5. The local receptive field for each 3 X 3 filters is 9 while global receptive field for second one is 5 X 5.

This is preffered as 5 X 5 requires 25 computations while two 3 X 3 require 9 + 9 = 18 computations. 
