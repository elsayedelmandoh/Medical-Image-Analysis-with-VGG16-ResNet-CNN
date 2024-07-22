# Medical Image Analysis with CNN

## Table of Contents
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
  - [Chest X-Ray Images](#chest-x-ray-images)
  - [NIH Clinical Center](#nih-clinical-center)
- [Code Structure](#code-structure)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Author](#author)


## Project Objectives
In this project, you will build Convolutional Neural Network (CNN), VGG16, and ResNet models for medical image analysis, such as X-rays or MRIs, for detecting diseases or abnormalities. The goal is to analyze medical images, such as X-rays or MRIs, to detect diseases or abnormalities. 

1. **Exploratory Data Analysis (EDA):** Analyze the dataset, explore correlations between features, and handle outliers or missing values.

2. **Preprocessing:** Preprocess the dataset by splitting it into training and testing sets. Normalize pixel values to prepare for model training.

3. **Build CNN Model:** Utilize TensorFlow and Keras to build a CNN model. Experiment with different architectures, activation functions, and learning rates to find the optimal model for medical image analysis.

4. **Model Evaluation:** Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score. Visualize the results with confusion matrices.


## Dataset

### Chest X-Ray Images
- Dataset Link: [Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Number of Images: 5856
- Classes: Normal, Pneumonia

### NIH Clinical Center
- Dataset Link: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Number of Images: 100,000+
- Disease Labels: 14 different labels

## Code Structure

The code is organized into different sections:
- **Verify GPU**: Check the availability of GPU.
- **Data Acquisition**: Download and preprocess the dataset using TensorFlow and OpenCV.
- **Data Exploration and Preprocessing**: Explore the dataset and preprocess it for training.
- **EDA (Exploratory Data Analysis)**: Visualize the distribution of classes in the dataset.
- **CNN Model Architecture and Hyperparameter Tuning**: Build and optimize a CNN model using scikeras and RandomizedSearchCV.
- **VGG16 Model Architecture and Hyperparameter Tuning**: Build and optimize a VGG16 model.
- **ResNet Model Architecture and Hyperparameter Tuning**: Build and optimize a ResNet model.
- **Callbacks**: Implement callbacks such as ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping.
- **Training the Model**: Train the models with the prepared dataset.
- **Displaying Metrics**: Visualize the training and validation metrics.
- **Evaluation Metrics for the Test Set**: Evaluate the models on the test set and display classification reports and confusion matrices.
- **Model Comparison**: Compare the performance of different models

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/elsayedelmandoh/Medical-Image-Analysis-with-VGG16-ResNet-CNN.git
    ```
   ```bash
   cd Medical-Image-Analysis-with-CNN
    ```
   
## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!
