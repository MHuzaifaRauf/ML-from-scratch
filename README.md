# K-Nearest Neighbors (KNN) Implementation for Abalone Age Prediction

## Project Overview

This project implements a K-Nearest Neighbors (KNN) algorithm from scratch to predict the age of abalone based on physical measurements. The implementation includes finding the optimal K value, training the model, and evaluating its performance on both training and test sets.

## Authors

- Muhammd Huzaifa Rauf (211370180)
- Ammad-Ud-Din (211370187)
- Abdul Mateen (211370194)
- Abdul Rehman Khan (211370196)

ML-Section-A

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib

You can install these libraries using pip:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## How to Run the Code

1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries as mentioned above.
3. Download the `KNN.ipynb` file.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run all cells in the notebook sequentially.

## What the Code Does

1. **Data Loading and Preprocessing**:

   - Loads the Abalone dataset from UCI Machine Learning Repository.
   - Drops the 'Sex' column as it doesn't impact the age prediction.
   - Splits the data into features (X) and target variable (y).

2. **KNN Implementation**:

   - Implements a custom KNN class with fit and predict methods.

3. **Model Evaluation**:

   - Implements functions to calculate accuracy, precision, recall, and F1-score.
   - Finds the best K value by testing a range of K values (1 to 99, odd numbers only).

4. **Results Analysis**:

   - Prints metrics for each K value tested.
   - Identifies and reports the best K value.
   - Trains the model with the best K and reports performance on both training and test sets.

5. **Visualization**:

   - Plots the relationship between K values and accuracies (both training and test).

6. **Results Saving**:
   - Saves the results for each K value to a text file named 'k_values.txt'.

## Interpreting the Results

1. **Best K Value**:

   - The program will output the best K value, which gives the highest test accuracy.

2. **Performance Metrics**:

   - For each K value tested, you'll see:
     - Training Accuracy
     - Test Accuracy
     - Precision
     - Recall
     - F1-Score

3. **Final Model Performance**:

   - After determining the best K, the program will show the final model's performance on both training and test sets.

4. **K vs. Accuracy Plot**:

   - A plot will be generated showing how training and test accuracies change with different K values.

5. **k_values.txt File**:
   - This file contains detailed results for each K value tested, which can be useful for further analysis.

## Understanding the Output

- Higher accuracy, precision, recall, and F1-score indicate better model performance.
- If training accuracy is significantly higher than test accuracy, it might indicate overfitting.
- The best K value balances between overfitting (low K) and underfitting (high K).
- The plot helps visualize how model performance changes with different K values.

By analyzing these results, you can understand how well the KNN model performs in predicting abalone age and how different K values affect its performance.
