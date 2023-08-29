# Handwritten-Digit-Classifier
Classifier trained to predict handwritten digits with a GUI to write digits for prediction

The classifier is a Convolutional Neural Network (CNN) trained using TensorFlow's Keras API. The goal of this project was to develop an understanding of CNNs for image processing and to effectively implement data augmentation such that a robust classifier can be trained. 

## The Model:
 - CNN
   - Input layer: 28x28x1 greyscale images
   - Conv2d 32 filters with 3x3 kernel
   - Conv2d 64 filters with 3x3 kernel
   - Conv2d 128 filters with 3x3 kernel
   - Dense layer with 256 units
   - Output with 10 units
   - All layers used Leaky ReLu activations except for the output layer which passed through a Softmax activation
- Regularization was implemented using dropout layers


The model was trained for 25 epochs using the MNIST dataset along with random translations of half the dataset. Batch Normalization was used to stabilize training. The loss function used to train the model was categorical cross-entropy with Adam optimization.

Here's a short video of the final model predicting digits in the GUI:


https://github.com/steez-ml/Handwritten-Digit-Classifier/assets/39159387/65d838c8-489e-4d75-85ed-15783b2b8b01

