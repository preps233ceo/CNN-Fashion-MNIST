# CNN-Fashion-MNIST
CNN Fashion MNIST Classifier
This project uses a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset contains 70,000 grayscale images of 28x28 pixels, divided into 10 fashion categories. The goal is to build a model that can accurately classify these images.

Table of Contents
Installation
Usage
Model Architecture
Training
Evaluation
Prediction
Contributing
License
Installation
To run this project, you need to have Python installed along with several packages. You can install the required packages using pip:

sh
Copy code
pip install tensorflow numpy matplotlib
Usage
Clone this repository:
sh
Copy code
git clone https://github.com/yourusername/cnn-fashion-mnist.git
cd cnn-fashion-mnist
Run the Python script:
sh
Copy code
python fashion_mnist_classifier.py
Model Architecture
The CNN model is built using TensorFlow and Keras. The architecture consists of the following layers:

Conv2D: 32 filters, kernel size (3, 3), ReLU activation, input shape (28, 28, 1)
MaxPooling2D: Pool size (2, 2)
Conv2D: 64 filters, kernel size (3, 3), ReLU activation
MaxPooling2D: Pool size (2, 2)
Conv2D: 128 filters, kernel size (3, 3), ReLU activation
Flatten
Dense: 128 neurons, ReLU activation
Dropout: 50% dropout rate
Dense: 10 neurons, softmax activation
Training
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. It is trained for 10 epochs with a batch size of 64. 20% of the training data is used for validation during training.

python
Copy code
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
Evaluation
After training, the model is evaluated on the test set to measure its performance. The accuracy on the test set is printed.

python
Copy code
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
Prediction
The script includes a function to make predictions on a couple of sample images from the test set. It displays the images along with their actual and predicted labels.

python
Copy code
def predict_images(images, labels):
    predictions = model.predict(images)
    for i in range(len(images)):
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Actual: {labels[i]}, Predicted: {np.argmax(predictions[i])}')
        plt.show()
Contributing
Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.

Fork the repository
Create your feature branch (git checkout -b feature/fooBar)
Commit your changes (git commit -am 'Add some fooBar')
Push to the branch (git push origin feature/fooBar)
Create a new Pull Request
