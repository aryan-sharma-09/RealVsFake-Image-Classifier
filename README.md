# Real vs Fake Image Detection using CNN & PyTorch

This project helps classify images as **Real** or **Fake** using a custom Convolutional Neural Network (CNN) built with **PyTorch**.

## What the Project Does

The project shows the deep learning workflow. This includes:

- Preparing images for training

- Building a CNN model

- Training the model

- Checking how the model works

- Saving the model

- Using the model to classify new images

## Key Features

*   A custom CNN built from scratch

*   Preparing images using **Torchvision**

*   A system for training and checking the model

*   Saving the model at points

*   Using the trained model to classify images

*   Inputting the path to an image

*   Getting a confidence score for the classification

## Technologies Used

*   **Python**

*   **PyTorch**

*   **Torchvision**

*   **Pillow (PIL)**

*   **. Github**

*   **VS Code**

## Project Structure

RealVsFake/


## 📂 Project Structure

```text
RealVsFake/
│
├── sample_images/
│   ├── real.jpg
│   └── fake.jpg
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── rvf_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```


## Dataset

This project used the "140K Fake Faces Dataset".
The dataset is not included here because its very large.
You can download it from Kaggle:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces


## CNN Architecture

The CNN has these layers:

*   A **Convolution Layer**
*   **ReLU Activation**
*   **Max Pooling**
*   Another **Convolution Layer**
*   **ReLU Activation**
*   **Max Pooling**
*   A **Flatten Layer**
*   A Connected Layer
*   An Output Layer that says **Real** or **Fake**

## How to Train

Run:
python src/train.py

## How to Predict

Run:
python src/predict.py

Enter the image path when prompted.

For example:
sample_images/fake.jpg

or

sample_images/real.jpg

## Results

*   Training Accuracy: **~99%**

*   Validation Accuracy: **~93%**

The model works well on images, like those it was trained on.

## Limitations

The model was trained on **140K Fake Faces** images.
It works best on images and may not work perfectly on new types of images.

## Future Improvements

*   Using a -trained model like **ResNet** or **EfficientNet**
*   A larger and more varied dataset
*   Adding data variations
*   Creating a web app with **Streamlit**
*   Deploying the model online

## Author

**Aryan Sharma**

B.Tech Artificial Intelligence & Machine Learning

Symbiosis Institute of Technology ,Pune

GitHub:
https://github.com/aryan-sharma-09