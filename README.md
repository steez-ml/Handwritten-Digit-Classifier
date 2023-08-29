# Handwritten-Digit-Classifier
Classifier trained to predict handwritten digits with a GUI to write digits for prediction

The classifier is a Convolutional Neural Network (CNN) trained using TensorFlow's Keras API. The goal of this project was to develop an understanding of CNNs for image processing and to effectively implement data augmentation such that a robust classifier can be trained. 

##The Model:
 - CNN
   - Input layer: 28x28x1 greyscale images
   - Conv2d 32 filters with 3x3 kernel
   - Conv2d 64 filters with 3x3 kernel
   - Conv2d 128 filters with 3x3 kernel
   - Dense layer with 256 units
   - Output with 10 units
   All layers used Leaky ReLu activations except for the output layer which passed through a Softmax activation
- Regularization was implemented using dropout layers
- 
