import pandas as pd
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from datetime import datetime 

# Step 1 - Data Analysis: Examine the dataset, check for class imbalances, and missing data
data = pd.read_csv(r'.\Datasets\Turkish.csv')

# Step 2 - Data Preprocessing: Clean, normalize, and vectorize the data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['Data'])
y = data['Gender']

# Step 3 - Train, Validation, Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4 - Model Selection: If you want to use a different model, replace it here and make adjustments
model = MultinomialNB()

# Step 5 - Hyperparameter Tuning (Optional)
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}  # Define the hyperparameter range
grid_search = GridSearchCV(model, param_grid, cv=5)  # Find the best hyperparameters with cross-validation
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
model = MultinomialNB(alpha=best_alpha)

# Step 6 - Model Training
model.fit(X_train, y_train)

# Step 7 - Model Evaluation: Calculate accuracy and other performance metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Step 8 - Model Saving
current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
joblib_model_filename = f'./Models/Joblib/{current_datetime}.joblib'
joblib.dump(model, joblib_model_filename)

# Step 9 - Cross Validation and Improvement (Optional): Evaluate your model more reliably using cross-validation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
mean_cross_val_score = cross_val_scores.mean()

print("Model Evaluation on the Dataset:")
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_rep)
print("Cross-Validation Results:")
print(f'Mean Cross-Validation Accuracy: {mean_cross_val_score:.2f}')

# Step 10 - Load the model using Joblib
model_path = joblib_model_filename
model = joblib.load(model_path)

# Step 11 - Specify a fixed input dimension
num_features = 10

# Step 12 - Specify the input data type to create the ONNX model
X_sample = np.zeros((1, num_features), dtype=np.float32)
initial_type = [('input', FloatTensorType([None, X_sample.shape[1]]))]

# Step 13 - Convert the model to ONNX format
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Step 14 - Save the ONNX model to a file
output_path = f'./Models/Onnx/{current_datetime}.onnx'

with open(output_path, 'wb') as f:f.write(onnx_model.SerializeToString())