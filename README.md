# Machine Learning Algorithms from Scratch

This repository contains implementations of fundamental machine learning algorithms built from scratch using Python. The implementations include Decision Trees, Logistic Regression, and K-Nearest Neighbors (KNN).

## Project Structure

```
├── decision_trees.py           # Decision Tree Classifier implementation
├── logistic_regression.py      # Logistic Regression implementation
├── logistic_decision.ipynb     # Comparison notebook for bank churn prediction
├── KNN/                        # KNN implementation for Abalone age prediction
└── requirements.txt
```

## Implemented Algorithms

### 1. Bank Customer Churn Prediction

Located in the root directory, using:

- Decision Trees (`decision_trees.py`)
- Logistic Regression (`logistic_regression.py`)

Both algorithms are evaluated and compared in logistic_decision.ipynb using the Bank Customer Churn Prediction dataset from Kaggle.

Results from the comparison:

- Logistic Regression Accuracy: 80.35%
- Decision Tree Accuracy: 85.75%

### 2. Abalone Age Prediction

Located in the KNN directory:

- Implementation of K-Nearest Neighbors algorithm
- Uses UCI Machine Learning Repository's Abalone dataset
- Detailed documentation available in README.md

## Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. For Bank Churn Prediction:

   ```python
   from decision_trees import DecisionTreeClassifier
   from logistic_regression import LogisticRegression

   # Initialize models
   dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
   lr_model = LogisticRegression(learning_rate=0.0001, n_iters=1000)
   ```

2. For Abalone Age Prediction:
   - Navigate to the KNN directory
   - Follow instructions in README.md

## Model Parameters

### Decision Tree

- `max_depth`: Maximum depth of the tree (default=5)
- `min_samples_split`: Minimum samples required to split a node (default=2)

### Logistic Regression

- `learning_rate`: Step size for gradient descent (default=0.0001)
- `n_iters`: Number of iterations (default=1000)

### KNN

- See README.md in KNN directory for detailed parameters and usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.
