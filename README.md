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

This project involves building a Convolutional Neural Network (CNN) model to analyze medical images, such as X-rays or MRIs, for detecting diseases or abnormalities. The goal is to develop a robust model that can accurately classify images into binary classes: normal or pneumonia.

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

- **Data Acquisition:** Load and explore the dataset.
- **Data Visualization:** Visualize sample images from the training, validation, and test sets.
- **Data Preprocessing:** Preprocess the dataset, split into training and testing sets, and normalize pixel values.
- **CNN Model Architecture:** Build an advanced CNN model with various convolutional and dense layers.
- **Callbacks:** Implement callbacks for model checkpointing, learning rate reduction, and early stopping.
- **Model Training:** Train the CNN model using the prepared data.
- **Metrics and Visualization:** Display accuracy and loss metrics during model training. Evaluate the model's performance and visualize results.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/elsayedelmandoh/Medical-Image-Analysis-with-CNN.git
   
   ```bash
   cd Medical-Image-Analysis-with-CNN

## Contributing
  Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback.

## Author
  Elsayed Elmandoh : [Linkedin](https://www.linkedin.com/in/elsayed-elmandoh-77544428a/).
