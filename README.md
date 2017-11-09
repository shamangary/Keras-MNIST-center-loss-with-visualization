# Keras-MNIST-center-loss-with-visualization


<figure> <img src="https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization/blob/master/images/softmax_only/epoch%3D29.jpg" height="300"/> <figcaption> Fig1. Softmax only.</figcaption> </figure> <figure> <img src="https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization/blob/master/images/centerloss/epoch%3D29.jpg" height="300"/> <figcaption> Fig2. Softmax with center loss.</figcaption> </figure>


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


## References:
+ https://github.com/keunwoochoi/keras_callbacks_example
+ https://github.com/jxgu1016/MNIST_center_loss_pytorch
+ http://kexue.fm/archives/4493/
