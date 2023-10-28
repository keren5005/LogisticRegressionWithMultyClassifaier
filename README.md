# Project Name: Multiclass Logistic Regression with One-vs-All Approach

## Overview:
This project demonstrates the implementation of multiclass logistic regression using the one-vs-all approach. The goal is to classify data into multiple classes by training binary classifiers for each class and making predictions based on the class with the highest probability. The project includes implementations of the OneVsAll class, Logistic Regression, Ordinal Encoder, and Min-Max Scaler, as well as utility functions for data preprocessing and evaluation.

## Project Structure:

- **`OneVsAllClass.py`**: Contains the `OneVsAll` class implementation for multiclass logistic regression.
- **`LogisticRegressionClass.py`**: Defines the `LogReg` class for binary logistic regression.
- **`OrdinalEncoder.py`**: Provides the `OrdinalEncoder` class for mapping categorical variables to ordinal integers.
- **`MinMaxScalerClass.py`**: Implements the `MinMaxScaler` class for feature scaling.
- **`UtilityFunctions.py`**: Contains utility functions such as `print_confusion_matrix`, `load_and_scale`, and `train_test_split` for data manipulation and evaluation.
- **`hsbdemo.csv`**: Sample dataset used for demonstrating the multiclass logistic regression.

## Usage:

To run the project, follow these steps:

1. Clone the repository to your local machine:
   ```
   git clone <repository-url>
   ```
   
2. Ensure you have Python installed on your system.

3. Navigate to the project directory:
   ```
   cd multiclass-logistic-regression
   ```

4. Run the main script:
   ```
   python main.py
   ```

## Dependencies:

- **Python**: The project is implemented in Python.
- **NumPy**: Required for numerical operations and array manipulations.
- **Pandas**: Used for data manipulation and handling datasets.
- **Matplotlib**: Optional, used for data visualization in some scripts.

