# jester-project
Final project for Machine Learning 2

In this project, we use 3D ResNets to classify videos (image sequences) of humans performing various hand gestures.

The Jester dataset contains 148,092 videos of humans making one of 27 different hand gestures in front of a webcam. The videos are stored as directories of image sequences. The dataset, and more information about it, is available [here](https://20bn.com/datasets/jester/v1).

ResNets are a type of neural network that consists of 3D convolution and pooling layers. The most notable twist to ResNets is that the input (or "residual") is added back into the output, just before the ReLU() function at the end of each block. To learn more about ResNets, check out [this paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Hara_Learning_Spatio-Temporal_Features_ICCV_2017_paper.pdf) and [this corresponding code](https://github.com/kenshohara/3D-ResNets-PyTorch).