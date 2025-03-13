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
- `requirements.txt`: A list of all Python dependencies
- `hyperparameter_search_results.csv`: A CSV file containing the architectures and performance of every randomly generated CNN


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
## Methodology

### Data Preprocessing
1. **Data Cleaning:** Non-defective wafer maps are removed from the dataset.
2. **Resizing:** Wafer maps are resized to a consistent shape of (39, 39) using bilinear interpolation.
3. **Data Augmentation:** To address class imbalance, transformations like rotation, flipping, and zooming are applied to underrepresented classes.

### Classification Methods
1. **Naive Bayes:** Initially explored as a baseline, achieving ~51.6% accuracy.
2. **Convolutional Neural Network (CNN):** A 3-layer CNN was implemented and optimized using random hyperparameter search.

### CNN Hyperparameter Optimization
The following hyperparameters were explored:
- Filter Sizes: (32, 64, 128), (64, 128, 256)
- Kernel Sizes: ((3, 3), (3, 3), (3, 3)), ((5, 5), (3, 3), (3, 3))
- Activation Functions: relu, tanh, sigmoid
- Pooling Types: Max, Average
- Dropout Rates: 0.2, 0.5, 0.7
- Dense Units: 128, 256

### Results
The results of the hyperparameter search, including test accuracy and other performance metrics, are detailed in the `hyperparameter_search_results.csv` file. Ablation tests reveal ReLU and average pooling are the best configurations for test accuracy.

## Usage Guide
1. Ensure all required libraries are installed.
3. Open `main.ipynb` in Jupyter Notebook or any compatible environment.
4. Load the WM-811K dataset using the provided code.
5. Run the code for each of the following sections in the notebook:
  - System set-up: Import libraries.
  - Data Acquisition: Import WM-811K data from Kaggle.
  - Data Preprocessing: Load and preprocess the data.
  - Classification Methods: Run the baseline Naive Bayes classifier and Convolutional Neural Network models.
  - Results: Evaluate model performance and review visualizations.

## Future Work
- Implement parameter optimization techniques such as Bayesian optimization.
- Explore adding attention layers, such as self-attention or spatial attention, to the CNN architecture.
- Implement a grid search to systematically evaluate all possible combinations of hyperparameters.
- Investigate deeper CNN architectures or pre-trained models (e.g., ResNet, EfficientNet) to leverage transfer learning for better performance.
