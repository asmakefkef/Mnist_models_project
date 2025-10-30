# MNIST Neural Networks Comparative Study

## Project Overview
This project is a comparative study of two neural network architectures for classifying handwritten digits from the MNIST dataset. The two models implemented are:

1. **Convolutional Neural Network (CNN)**
2. **Long Short-Term Memory (LSTM)**

The goal is to compare their performance in terms of **accuracy**, **loss**, and **training time**.

---

## Dataset and Preprocessing
- **Dataset:** MNIST handwritten digits dataset (60,000 training images, 10,000 test images, 28x28 grayscale).
- **Preprocessing steps:**
  - Normalize pixel values to range [0,1].
  - Reshape images according to model requirements:
    - CNN: (28,28,1)
    - LSTM: (28,28)
  - Convert labels to one-hot encoding.

---

## Project Structure

Mnist_models_project/
├─ Src/
│ ├─ main.py # Main script to train and compare models
│ ├─ models/
│ │ ├─ cnn_model.py # CNN architecture and training function
│ │ └─ lstm_model.py # LSTM architecture and training function
│ └─ utils/
│ ├─ data_preprocessing.py # Functions for loading and preprocessing MNIST
│ └─ plot_utils.py # Functions to plot and save Accuracy/Loss comparison
├─ Results/
│ ├─ Accuracy_Comparison.png # Plot comparing CNN vs LSTM Accuracy
│ └─ Loss_Comparison.png # Plot comparing CNN vs LSTM Loss
└─ requirements.txt


---

## Training and Evaluation

Both models were trained for **5 epochs** with a **batch size of 128**.
**Training times on Colab:**
- CNN: ~340 seconds
- LSTM: ~185 seconds

**Performance:**
- CNN achieved slightly higher accuracy and lower loss on the validation set.
- LSTM performed well but required less training time.

**Plots saved in `Results/` folder:**
- `Accuracy_Comparison.png` → shows training & validation accuracy for both models.
- `Loss_Comparison.png` → shows training & validation loss for both models.

---

## How to Run

1. Install required packages:

```bash
pip install -r requirements.txt


Run the main script:

python Src/main.py


This will train both models, display the plots, and save them in the Results/ folder.

Discussion

CNN vs LSTM:

CNN is more efficient for image classification due to convolutional layers that capture spatial patterns.

LSTM can handle sequences and temporal dependencies but is less efficient for static image data like MNIST.

Training time: CNN took longer due to convolution operations, while LSTM was faster.

Improvements / Extensions:

Try deeper CNN or stacked LSTM layers.

Experiment with Transformers for image classification.

Apply data augmentation to improve generalization.

Tune hyperparameters (learning rate, batch size, number of epochs).
