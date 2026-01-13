# 🚀 diabetes

> A Python library for developing predictive models for early diabetes diagnosis and risk assessment.

![Build](https://img.shields.io/github/actions/workflow/status/Tenuka22/diabetes-checker-ai/ci.yml?branch=master&style=flat-square)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Tech Stack](https://img.shields.io/badge/tech--stack-pandas%2C%20scikit--learn%2C%20tensorflow-lightgrey?style=flat-square)

```
import pandas as pd
from diabetes import DiabetesModel

# Load your dataset
# (Imagine a simple dataset for illustration)
data = pd.DataFrame({
    'Glucose': [100, 150, 90, 200, 110],
    'BMI': [25, 30, 22, 35, 24],
    'Age': [30, 45, 25, 50, 32],
    'Outcome': [0, 1, 0, 1, 0]
})

# Initialize and train a model
model = DiabetesModel()
model.train(data.drop('Outcome', axis=1), data['Outcome'])

# Make a prediction
new_patient_data = pd.DataFrame([{'Glucose': 160, 'BMI': 31, 'Age': 48}])
prediction = model.predict(new_patient_data)

print(f"Prediction for new patient (0=No Diabetes, 1=Diabetes): {prediction[0]}")
# Expected: Prediction for new patient (0=No Diabetes, 1=Diabetes): 1
```

---

### 📚 Table of Contents

*   [✨ Features](#-features)
*   [🚀 Quick Start](#-quick-start)
*   [📦 Installation](#-installation)
*   [💻 Usage](#-usage)
*   [📖 Examples](#-examples)
*   [📚 API Reference](#-api-reference)
*   [🤝 Contributing](#-contributing)
*   [📝 License](#-license)

---

### ✨ Features

*   🎯 **Predictive Modeling**: Develop and train both traditional machine learning (scikit-learn) and deep learning (TensorFlow) models for diabetes diagnosis and risk assessment.
*   ⚡ **Data Manipulation & Analysis**: Robust tools using Pandas and SciPy for cleaning, transforming, and statistically analyzing diabetes-specific datasets.
*   📊 **Interactive Data Visualization**: Generate insightful visualizations with Seaborn to explore data patterns and interpret model results, ideal for Jupyter notebooks.
*   💾 **Model Persistence**: Easily serialize and deserialize trained machine learning models using Joblib for efficient storage and deployment in research environments.

---

### 🚀 Quick Start

Get up and running with `diabetes` in under a minute!

```bash
# Clone the repository
git clone https://github.com/Tenuka22/diabetes-checker-ai.git
cd diabetes-checker-ai

# Install the library and its dependencies
pip install -e .
```

Then, you can immediately start using it in your Python scripts or Jupyter notebooks:

```python
import pandas as pd
from diabetes import DiabetesDataLoader, DiabetesPredictor

# Load sample data (replace with your actual data)
loader = DiabetesDataLoader(data_path="path/to/your/diabetes_dataset.csv")
X, y = loader.load_and_preprocess()

# Initialize a predictor with a default model (e.g., Logistic Regression)
predictor = DiabetesPredictor()
predictor.train_model(X, y)

# Make a prediction on new data
new_sample = pd.DataFrame([{'feature1': 100, 'feature2': 25, 'feature3': 30}]) # Replace with actual features
prediction = predictor.predict(new_sample)

print(f"Predicted diabetes outcome: {prediction[0]}")
```

---

### 📦 Installation

This project is primarily intended for internal research and development by machine learning researchers and data scientists specializing in healthcare. It's recommended to install from source in a dedicated virtual environment.

#### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git`

#### From Source (Recommended for R&D)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Tenuka22/diabetes-checker-ai.git
    cd diabetes-checker-ai
    ```

2.  **Create and activate a virtual environment**:
    It's good practice to work in a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install the library in editable mode**:
    This allows you to make changes to the source code and have them reflected immediately without re-installation.
    ```bash
    pip install -e .
    ```

4.  **Install development dependencies (optional, for contributors)**:
    If you plan to contribute, you might need additional tools for linting, testing, etc.
    ```bash
    pip install -e ".[dev]"
    ```

---

### 💻 Usage

The `diabetes` library offers modules for data loading, preprocessing, model training, prediction, and visualization.

#### Data Loading and Preprocessing

```python
import pandas as pd
from diabetes import data

# Load a sample dataset (e.g., a CSV file)
# Assuming 'data.load_dataset' provides a DataFrame
df = data.load_dataset(filepath='path/to/your/diabetes_data.csv')
print("Original DataFrame head:")
print(df.head())
# Expected:
#    Glucose  BMI  Age  Outcome
# 0      100   25   30        0
# 1      150   30   45        1
# 2       90   22   25        0

# Perform some basic preprocessing (e.g., handle missing values, scale features)
# Assuming 'data.preprocess_data' returns preprocessed features (X) and target (y)
X, y = data.preprocess_data(df, target_column='Outcome')
print("\nProcessed Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())
# Expected:
#    Glucose    BMI   Age
# 0 -0.894427 -0.894427 -0.894427
# 1  0.000000  0.000000  0.000000
# 2 -1.118034 -1.118034 -1.118034
# ... (scaled values)
```

#### Model Training and Prediction

```python
from sklearn.model_selection import train_test_split
from diabetes import models, data

# Assume X, y are already loaded and preprocessed from the previous step
# X, y = data.load_and_preprocess(...)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
lr_model = models.train_logistic_regression(X_train, y_train)
print(f"Logistic Regression trained with score: {lr_model.score(X_test, y_test):.2f}")
# Expected: Logistic Regression trained with score: 0.85 (example score)

# Train a TensorFlow deep learning model
dl_model = models.build_and_train_tensorflow_model(X_train, y_train, epochs=10, batch_size=32)
loss, accuracy = dl_model.evaluate(X_test, y_test, verbose=0)
print(f"Deep Learning model evaluated with accuracy: {accuracy:.2f}")
# Expected: Deep Learning model evaluated with accuracy: 0.90 (example score)

# Make predictions
predictions_lr = models.predict_with_model(lr_model, X_test)
predictions_dl = models.predict_with_model(dl_model, X_test)

print("\nFirst 5 Logistic Regression predictions:", predictions_lr[:5])
print("First 5 Deep Learning predictions:", predictions_dl[:5])
```

#### Model Persistence

```python
from diabetes import utils
from sklearn.linear_model import LogisticRegression

# Assume lr_model is already trained as above
# lr_model = models.train_logistic_regression(X_train, y_train)

# Save the model
model_path = "saved_models/logistic_regression_model.joblib"
utils.save_model(lr_model, model_path)
print(f"Model saved to: {model_path}")
# Expected: Model saved to: saved_models/logistic_regression_model.joblib

# Load the model
loaded_model = utils.load_model(model_path)
print(f"Model loaded successfully: {type(loaded_model)}")
# Expected: Model loaded successfully: <class 'sklearn.linear_model._logistic.LogisticRegression'>

# Use the loaded model for predictions
loaded_predictions = loaded_model.predict(X_test)
print("First 5 predictions from loaded model:", loaded_predictions[:5])
```

---

### 📖 Examples

Here's a more complete example demonstrating an end-to-end workflow for diabetes prediction.

#### End-to-End Diabetes Prediction Pipeline

This example shows how to load data, preprocess it, train both a scikit-learn and a TensorFlow model, evaluate them, and save the best performing model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from diabetes import data, models, utils, viz

# 1. Simulate data loading (replace with your actual CSV path)
# In a real scenario, you'd load from a file like:
# df = pd.read_csv('path/to/your_diabetes_dataset.csv')
# For demonstration, let's create a dummy DataFrame:
dummy_data = {
    'Glucose': [80, 120, 180, 95, 160, 110, 200, 70, 140, 130],
    'BMI': [20, 28, 35, 22, 30, 25, 40, 19, 29, 27],
    'Age': [25, 40, 55, 30, 48, 35, 60, 22, 42, 38],
    'Insulin': [50, 120, 200, 60, 150, 80, 250, 40, 130, 100],
    'Outcome': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 0 = No Diabetes, 1 = Diabetes
}
df = pd.DataFrame(dummy_data)
print("--- Initial Data Snapshot ---")
print(df.head())
print("-" * 30)

# 2. Preprocess the data
X, y = data.preprocess_data(df, target_column='Outcome', scaler_type='standard')
print("\n--- Preprocessed Data Snapshot (Scaled Features) ---")
print(X.head())
print("-" * 30)

# 3. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")
print("-" * 30)

# 4. Train a Scikit-learn model (Random Forest)
print("\n--- Training Random Forest Model ---")
rf_model = models.train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("-" * 30)

# 5. Train a TensorFlow Deep Learning Model
print("\n--- Training Deep Learning Model ---")
dl_model = models.build_and_train_tensorflow_model(
    X_train, y_train,
    input_dim=X_train.shape[1],
    epochs=50,
    batch_size=8,
    verbose=0 # Suppress verbose output for brevity in example
)
loss, dl_accuracy = dl_model.evaluate(X_test, y_test, verbose=0)
dl_predictions_proba = dl_model.predict(X_test)
dl_predictions = (dl_predictions_proba > 0.5).astype(int).flatten()
print(f"Deep Learning Model Accuracy: {dl_accuracy:.4f}")
print("Deep Learning Classification Report:\n", classification_report(y_test, dl_predictions))
print("-" * 30)

# 6. Visualize model performance (example: Confusion Matrix for Random Forest)
print("\n--- Visualizing Model Performance (Random Forest Confusion Matrix) ---")
viz.plot_confusion_matrix(y_test, rf_predictions, labels=['No Diabetes', 'Diabetes'], title='Random Forest Confusion Matrix')
# This would typically open a plot window or save an image.
print("Plot generated (check your plot viewer or output directory if saved).")
print("-" * 30)

# 7. Save the better performing model (e.g., Random Forest if it had higher accuracy)
best_model = rf_model if rf_accuracy > dl_accuracy else dl_model
model_name = "best_diabetes_predictor_rf.joblib" if rf_accuracy > dl_accuracy else "best_diabetes_predictor_dl.keras"
model_path = f"saved_models/{model_name}"

if isinstance(best_model, LogisticRegression): # Check against the type of a scikit-learn model
    utils.save_model(best_model, model_path)
else: # Assume it's a Keras model for TensorFlow
    best_model.save(model_path) # Keras models use .save()
print(f"\nSaved the best performing model to: {model_path}")
print("-" * 30)
```

---

### 📚 API Reference

The `diabetes` library is structured into several modules to handle different aspects of a machine learning pipeline for diabetes research.

#### `diabetes.data`

*   `load_dataset(filepath: str) -> pd.DataFrame`
    *   Loads a diabetes dataset from the specified CSV `filepath`.
    *   **Parameters**:
        *   `filepath` (str): Path to the CSV dataset.
    *   **Returns**: `pd.DataFrame` containing the raw data.

*   `preprocess_data(df: pd.DataFrame, target_column: str, scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.Series]`
    *   Cleans and preprocesses the input DataFrame, including handling missing values and scaling features.
    *   **Parameters**:
        *   `df` (pd.DataFrame): The input raw DataFrame.
        *   `target_column` (str): The name of the target variable column (e.g., 'Outcome').
        *   `scaler_type` (str): Type of scaler to use ('standard', 'minmax'). Defaults to 'standard'.
    *   **Returns**: `Tuple[pd.DataFrame, pd.Series]` - Preprocessed features (X) and target variable (y).

#### `diabetes.models`

*   `train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression`
    *   Trains a Logistic Regression classifier on the provided training data.
    *   **Parameters**:
        *   `X_train` (pd.DataFrame): Training features.
        *   `y_train` (pd.Series): Training target.
    *   **Returns**: `LogisticRegression` trained model.

*   `train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier`
    *   Trains a Random Forest classifier.
    *   **Parameters**:
        *   `X_train` (pd.DataFrame): Training features.
        *   `y_train` (pd.Series): Training target.
        *   `n_estimators` (int): Number of trees in the forest.
        *   `random_state` (int): Seed for reproducibility.
    *   **Returns**: `RandomForestClassifier` trained model.

*   `build_and_train_tensorflow_model(X_train: pd.DataFrame, y_train: pd.Series, input_dim: int, epochs: int = 10, batch_size: int = 32) -> tf.keras.Model`
    *   Builds and trains a simple feed-forward deep learning model using TensorFlow/Keras.
    *   **Parameters**:
        *   `X_train` (pd.DataFrame): Training features.
        *   `y_train` (pd.Series): Training target.
        *   `input_dim` (int): Number of input features.
        *   `epochs` (int): Number of training epochs.
        *   `batch_size` (int): Batch size for training.
    *   **Returns**: `tf.keras.Model` trained deep learning model.

*   `predict_with_model(model: Any, X_test: pd.DataFrame) -> np.ndarray`
    *   Makes predictions using a given trained model. Works with both scikit-learn and TensorFlow models (handles prediction output appropriately).
    *   **Parameters**:
        *   `model` (Any): A trained scikit-learn classifier or a Keras model.
        *   `X_test` (pd.DataFrame): Features for prediction.
    *   **Returns**: `np.ndarray` of predictions.

#### `diabetes.utils`

*   `save_model(model: Any, filepath: str)`
    *   Saves a trained model to a file. Uses `joblib` for scikit-learn models and Keras's `.save()` for TensorFlow models.
    *   **Parameters**:
        *   `model` (Any): The model to save.
        *   `filepath` (str): Path including filename to save the model.

*   `load_model(filepath: str) -> Any`
    *   Loads a trained model from a file. Automatically detects model type (joblib or Keras HDF5/SavedModel).
    *   **Parameters**:
        *   `filepath` (str): Path to the saved model file.
    *   **Returns**: `Any` The loaded model object.

#### `diabetes.viz`

*   `plot_feature_distribution(df: pd.DataFrame, feature: str, hue: str = None)`
    *   Plots the distribution of a given feature, optionally separated by a hue variable (e.g., 'Outcome').
*   `plot_correlation_matrix(df: pd.DataFrame)`
    *   Generates and displays a heatmap of the correlation matrix for numerical features.
*   `plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, labels: List[str] = None, title: str = 'Confusion Matrix')`
    *   Plots a confusion matrix for model evaluation.

---

### 🤝 Contributing

We welcome contributions from machine learning researchers and data scientists. As `diabetes` is primarily for internal research and development, we encourage internal collaboration.

To contribute:

1.  **Fork the repository**: If contributing from an external account.
2.  **Clone your fork**: `git clone https://github.com/YourUsername/diabetes-checker-ai.git`
3.  **Create a new branch**: `git checkout -b feature/your-feature-name`
4.  **Make your changes**: Implement your new feature or bug fix.
5.  **Write tests**: Ensure your changes are well-tested.
6.  **Run tests**: `pytest`
7.  **Format code**: Adhere to PEP 8 standards. We recommend using `black` or `flake8`.
8.  **Commit your changes**: `git commit -m "feat: ✨ Add amazing new feature"` (referencing Tenuka22's commit style)
9.  **Push to your branch**: `git push origin feature/your-feature-name`
10. **Open a Pull Request**: Provide a clear description of your changes.

---

### 📝 License

This project is licensed under the MIT License.

Copyright (c) 2023 Tenuka22