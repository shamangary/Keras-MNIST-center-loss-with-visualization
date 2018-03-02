# Keras-MNIST-center-loss-with-visualization


<img src="https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization/blob/master/images/softmax_only/epoch%3D49.jpg" height="300"/> <img src="https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization/blob/master/images/centerloss/epoch%3D49.jpg" height="300"/>

<center> Fig. (left) Softmax only. (right) Softmax with center loss </center> 

## Update (2018/03/02)
+ Code explanation in Chinese.
http://shamangary.logdown.com/posts/6424093

## Update (2017/11/10)
+ Remove the one-hot inputs for Embedding layer and replace it by single value labels.
+ There are two kinds labels: single value for center loss, and one-hot vector labels for softmax term.
+ Every classes are visually seperated now :)

## How to run?
+ Step.1
Change the flag of center loss inside TYY_mnist.py
```
isCenterloss = True
#isCenterloss = False
```
+ Step.2
Run the file
```
KERAS_BACKEND=tensorflow python TYY_mnist.py
```

## Dependencies
+ Anaconda
+ Keras
+ Tensorflow
+ Others: (install with anaconda)
```
conda install -c anaconda scikit-learn 
conda install -c conda-forge matplotlib
```


## References:
+ https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
+ https://github.com/keunwoochoi/keras_callbacks_example
+ https://github.com/jxgu1016/MNIST_center_loss_pytorch
+ http://kexue.fm/archives/4493/
