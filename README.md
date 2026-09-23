# Plant-Disease-Risk-Classification-
Machine learning-based plant disease classification system using Python, Scikit-learn, Random Forest, and Streamlit to classify plant conditions based on leaf and environmental features.
# Plant Disease Classification Using Machine Learning

## Project Description

Plant Disease Classification is a machine learning project developed using Python and Scikit-learn to classify plant leaf conditions based on leaf characteristics and environmental parameters.

The project uses a Random Forest Classification algorithm to identify four different plant conditions:

* Healthy
* Powdery Mildew
* Leaf Spot
* Rust

The application provides an interactive Streamlit interface where users can enter plant and environmental information and receive a predicted disease class along with the model's confidence.

## Features

* Plant disease classification using Machine Learning
* Random Forest Classification algorithm
* Interactive Streamlit web application
* Disease prediction with confidence score
* Disease probability breakdown
* Leaf and environmental parameter analysis
* CSV dataset support
* Model performance evaluation
* Confusion matrix visualization
* Pre-trained machine learning model
* Command-line prediction support

## Input Features

The model uses the following seven features:

| Feature            | Description                                            |
| ------------------ | ------------------------------------------------------ |
| Leaf Color Score   | Indicates the color condition of the leaf from 1 to 10 |
| Leaf Spot Count    | Number of visible spots or lesions on the leaf         |
| Leaf Moisture      | Moisture level on the leaf surface                     |
| Leaf Texture Score | Indicates the texture condition of the leaf            |
| Plant Age Days     | Age of the plant in days                               |
| Soil Moisture      | Moisture level in the soil                             |
| Temperature        | Ambient temperature around the plant                   |

## Disease Classes

### Healthy

Represents plants with normal leaf characteristics and no major disease indicators.

### Powdery Mildew

Represents plants showing characteristics associated with powdery mildew.

### Leaf Spot

Represents plants with increased leaf spot or lesion characteristics.

### Rust

Represents plants with characteristics associated with rust disease.

## Machine Learning Algorithm

The project uses a **Random Forest Classifier** from Scikit-learn.

The model is configured with:

* 250 decision trees
* Balanced class weights
* Fixed random state for reproducibility
* Stratified train-test split
* 80% training data
* 20% testing data

Random Forest combines multiple decision trees to perform classification and determine the most likely disease class from the input features.

## Machine Learning Workflow

```text
Synthetic Plant Dataset
          |
          v
Data Preparation
          |
          v
Feature Selection
          |
          v
Train-Test Split
          |
          v
Random Forest Training
          |
          v
Model Evaluation
          |
          v
Model Serialization
          |
          v
Streamlit Application
          |
          v
Disease Prediction
```

## Dataset

The project contains a synthetic dataset named:

```text
data/plant_disease_data.csv
```

The dataset contains 720 generated samples, with 180 samples for each disease class.

The dataset includes:

* 7 input features
* 1 target variable
* 4 disease classes

The data is generated programmatically using NumPy and is intended for academic and educational purposes.

## Project Structure

```text
Plant_Disease_Classification_Sklearn/
│
├── app.py
├── predict.py
├── train_model.py
├── plant_disease_classifier.pkl
├── plant_disease_confusion_matrix.png
│
├── data/
│   └── plant_disease_data.csv
│
├── requirements.txt
└── README.md
```

## File Description

### app.py

Contains the Streamlit web application.

It provides:

* Interactive input controls
* Disease prediction
* Prediction confidence
* Probability distribution
* Disease information
* Treatment suggestions
* CSV-based field survey functionality
* Model performance visualization
* Confusion matrix display

### train_model.py

Responsible for:

* Generating the synthetic dataset
* Saving the dataset as CSV
* Splitting the data into training and testing sets
* Training the Random Forest model
* Evaluating the model
* Generating the confusion matrix
* Saving the trained model

### predict.py

Provides a simple Python script for testing the trained model with sample plant data.

### plant_disease_classifier.pkl

Contains the trained Random Forest classification model saved using Joblib.

### plant_disease_confusion_matrix.png

Contains the confusion matrix generated during model evaluation.

### requirements.txt

Contains the Python packages required to run the project.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Plant_Disease_Classification_Sklearn.git
```

Navigate to the project directory:

```bash
cd Plant_Disease_Classification_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

To generate the dataset and train the Random Forest model:

```bash
python train_model.py
```

This creates:

```text
data/plant_disease_data.csv
plant_disease_classifier.pkl
plant_disease_confusion_matrix.png
```

## Run Prediction

To test the trained model using the sample input:

```bash
python predict.py
```

The program displays:

```text
Plant Disease Classification
----------------------------
Predicted Class: ...
Confidence: ...%
```

## Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The Streamlit application provides an interactive interface for entering plant characteristics and obtaining disease predictions.

## Model Evaluation

The model is evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix

The classification report provides precision, recall, and F1-score for each disease class.

The confusion matrix is saved as:

```text
plant_disease_confusion_matrix.png
```

## Application Workflow

1. Enter leaf characteristics.
2. Enter environmental conditions.
3. Click the diagnostic button.
4. The trained Random Forest model processes the input.
5. The application predicts the most likely plant condition.
6. The prediction confidence is displayed.
7. Probabilities for all disease classes are shown.
8. Additional disease information and suggested treatment guidance are displayed.

## Important Note

This project uses a **synthetic dataset generated for educational purposes**. The predictions should not be considered a substitute for professional agricultural or plant pathology diagnosis.

The model is designed to demonstrate machine learning classification, data preparation, model training, evaluation, and deployment using Streamlit.

## Future Enhancements

* Use real plant leaf image datasets
* Implement image-based disease classification using CNNs
* Add more plant species and diseases
* Integrate real agricultural datasets
* Add image upload functionality
* Improve model validation using cross-validation
* Add model comparison with other algorithms
* Deploy the application online
* Add database support for storing predictions
* Add multilingual support

