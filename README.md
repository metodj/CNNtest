#PKP summary – Metod, Aneja collaboration

Our main task was to try and construct a neural network for 3D medical image segmentation. After initial literature review we decided to use an article from Prof. Dr. Patrick van der Smagt and his research group from Technische Universität München as a reference. They used a CNN-based (convolutional neural networks) method with three-dimensional filters and applied their algorithms on two public datasets (hand and brain MRI). The network was divided into several contracting and expanding stages with feature maps from different stages being combined together via long skip connections.

The authors of the paper used their own Python library Breze for the implementation of the proposed CNN. However, due to the lack of documentation, we decided to use an open source Python library for building and training neural networks, Lasagne. Although Lasagne proved to be very practical and easy to use, it was missing some of the components that we needed for handling 3D data. Therefore, we had to implement some of those by ourselves, like a 3D deconvolutional layer, our own activation function layer, etc. We also had some problems with dimension miss-matching (because of compressing and stretching of the volumes throughout our network), final aggregation of the feature maps into a segmentation map for prediction and acquiring the right data to do our training and testing. We contacted the authors of the paper to discuss some of our problems and in the end managed to get the network ready to use.

After the implementation, we first tried to train our network on our computers and later also on the cluster that we got access to. We first tried with publicly available data from the cancer imagining archive, CT scans of lungs acquired at different lungs' capacities. First results were not promising as the loss would not converge, which meant the network was not learning properly. We then also tried training on brain MRI images and played around with using different loss functions, tweaking the network's properties, using different activation functions, etc. in order to make the network to show some results, yet we were unable to solve the problems and make the network successful. It could do with our lack of knowledge or just some minor mistakes in network's configurations. 

The majority of the source code, except for some data pre-processing, is available on this repository. 
