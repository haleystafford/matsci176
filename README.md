# Wafer Map Classification Using a 3-layer Convolutional Neural Network

## Project Description
This project aims to automate the classification of defect patterns in semiconductor wafer maps using machine learning techniques. Wafer maps visually represent the state of semiconductor wafers, and detecting defects is critical for ensuring manufacturing quality. Using the publicly available WM811K dataset, we explore two classification methods:
1. **Naive Bayes**: A baseline method for quick evaluation of the dataset.
2. **Convolutional Neural Networks (CNNs):** A more advanced approach leveraging spatial hierarchies in image data.

The ultimate goal is to develop a CNN model capable of classifying defective wafers into one of eight failure type categories: “Scratch”, “Edge-Ring”, “Edge-Loc”, “Center”, “Random”, “Loc”, “Near-full”, and “Donut”.

The provided Jupyter notebook (`main.ipynb`) contains all the necessary code, explanations, utility functions, and results for replicating the analysis.

## Dataset
The project uses the WM811K dataset, which contains over 800,000 wafer maps with associated defect pattern labels. Each wafer map is a 2D grid where:
- 0: represents no die
- 1: represents a normal die
- 2: represents a defective die

### Data Source: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map


## Repository Structure
- `main.ipynb`: A comprehensive notebook that includes data loading, preprocessing, model training, evaluation and results visualization
- `README.md`: An overview of the project and setup instructions
- `requirements.txt file`: A list of all Python dependences
- `hyperparameter_search_results.csv`: A csv file containing the architectures and performance of every randomly-generated CNN


## Installation
To run the project, first clone this repository to your local machine:
``` bash
git clone https://github.com/haleystafford/matsci176
```

Second, install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Data Acquisition
The data for this project is accessible via KaggleHub. Install it using:
```bash
pip install kagglehub
```
To download the WM811K dataset from Kaggle, ensure you have KaggleHub installed. The following line in `main.ipynb` will download and extract the dataset automatically:
```bash
path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
```

### Data Preprocessing Steps in `main.ipynb`
1. **Data Cleaning:** Non-defective wafer maps are removed from the dataset.
2. **Resizing:** Wafer maps are resized to a consistent shape of (39, 39) using bilinear interpolation.
3. **Data Augmentation:** To address class imbalance, transformations like rotation, flipping, and zooming are applied to underrepresented classes.

## Usage Guide
1. Open `main.ipynb` in Jupyter Notebook or any compatible environment.
2. Follow these steps within the notebook:
-     Data Preprocessing: Load and preprocess the WM811K dataset.
-     Naive Bayes Classification: Run the baseline Naive Bayes classifier.
-     CNN Training: Train a 3-layer CNN with randomly sampled hyperparameters.
-     Results Analysis: Evaluate model performance and review visualizations.



## `main.ipynb` Walkthrough
**1. Data Preprocessing:** This section explains how to load, clean and preprocess the WM-811K dataset. Since the wafer maps do not have a consistent size we resized wafer maps using bilinear interpolation . This method stretches the wafer map while keeping features recognizable, creating slight distortions. We also performed data augmentation by increasing the sample size of each class to 75% of the size of the largest class.

**2. Classification Method 1 – Naive Bayes**: We initially explored  Naive Bayes as a baseline method for classifying the wafer maps. This model assumes that the features in the data (i.e. wafer maps) are independent and follow a Gaussian distribution. Although these assumptions might not perfectly hold for the wafer maps, Naive Bayes performed very well in classifying some failure types and terribly in others. This method was explored to evaluate the data before exploring more complex methods.

**3. Classification Method 2 – Convolutional Neural Network (CNN)**: In this section, we aimed to optimize a 3-layer CNN model to maximize classification accuracy. We performed random hyperparameter search on the following search space:

# Hyperparameter Search Space
filter_sizes_list = [(32, 64, 128), (64, 128, 256)]
kernel_sizes_list = [((3, 3), (3, 3), (3, 3)), ((5, 5), (3, 3), (3, 3))]
activation_list = ['relu', 'tanh', 'sigmoid']
pool_type_list = ['Max', 'Average']
dropout_rates = [0.2, 0.5, 0.7]
dense_units_list = [64, 128, 256]


**4. Results**: The code outputs the random hyperparameter search results in a csv file. Ablation tests reveal ReLU and average pooling are the best configurations for test accuracy. Future work should build models using these parameters and changing the other stuff to find the best combination of parameters for the 3-layer CNN.

