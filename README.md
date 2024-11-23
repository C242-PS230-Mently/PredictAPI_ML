# Machine Learning Mently
Repository for machine learning models
# Classification Stress Level : Mently App
This is the main repository for the final machine learning model to classify stress levels based on the answers to the questions. The stress level classification will focus on 5 mental health disorders:
1. Anxiety
2. Depression
3. Bipolar
4. Schizophrenia
5. OCD
## Overview
By using machine learning models, it will be easier to determine a person's stress level without the need to use the usual if-else logic again. and also this model will be very helpful for future updates. Users can also view their stress levels ranging from Low, Medium, and High.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Random Forest](#random-forest)
- [Results](#results)
- [License](#license)

## Getting Started

### Prerequisites
To install all the libraries needed in this model, use the syntax below:
```bash
pip install -r requirements.txt
```
### Installation
```bash
# Clone the repository
git clone https://github.com/C242-PS230-Mently/MachineLearning.git 

# Change directory
cd MachineLearning # change with your folder name

# Install dependencies
pip install -r requirements.txt
```
## Usage

In the process of making this model, there are several stages that are carried out and the use of the required data.

### Data Preparation
Prepare the dataset in a CSV file with the following structure:
Columns:
Q1 to Q25 (Answers to questions)
Level Anxiety, Level Depression, Level Bipolar, Level Schizophrenia, Level OCD (Target labels)

Answer Coding:
1 = Never, 2 = Sometimes, 3 = Often

Stress Level Calculation:
Each mental health condition is assessed through a subset of questions:

Q1-Q5 → Anxiety Level
Q6-Q10 → Depression Level
Q11-Q15 → Bipolar Level
Q16-Q20 → Schizophrenia Level
Q21-Q25 → OCD Level
Levels are categorized based on the sum of answers:

1-5 → Light (Level 1)
6-10 → Medium (Level 2)
11-15 → High (Level 3)

### Model Training
For the model training process, we used tensorflow.keras which uses 3 hidden layers. Which will be compiled using *adam* optimizer and loss params *binary_crossentropy*.
```bash
# Split dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Bagian 1: Latih model TensorFlow untuk setiap label ---
# Normalisasi data
X_train_norm = X_train / X_train.max()
X_test_norm = X_test / X_train.max()

# Definisi model TensorFlow untuk setiap label
tensorflow_models = {}
for column in y.columns:
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Menggunakan sigmoid untuk prediksi binary
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
   
```
### Model Evaluation
To evaluate the model, we first train it for 10 iterations or epochs. then we evaluate it based on the train and testing data.
```bash
 # Training model dengan 10 epoch
    model.fit(X_train_norm, y_train[column], epochs=10, batch_size=32, verbose=1)
    tensorflow_models[column] = model

# Prediksi dan evaluasi dengan TensorFlow
y_pred_tf = pd.DataFrame()
for column in y.columns:
    y_pred_tf[column] = tensorflow_models[column].predict(X_test_norm).flatten()

# Evaluasi model TensorFlow
for column in y.columns:
    print(f"Classification Report untuk {column} (TensorFlow):")
    print(classification_report(y_test[column], (y_pred_tf[column] > 0.5).astype(int)))
    print(f"Accuracy untuk {column} (TensorFlow): {accuracy_score(y_test[column], (y_pred_tf[column] > 0.5).astype(int))}\n")
# Simpan seluruh model TensorFlow dalam satu file
for column, model in tensorflow_models.items():
    model.save(f'tf_model_{column}.h5')
```

### Random Forest
We use the K-Means algorithm to cluster each data in the dataset.
```bash
# --- Bagian 2: Latih model Random Forest setelah TensorFlow ---
# Definisi model Random Forest untuk setiap label
rf_models = {}
for column in y.columns:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train[column])
    rf_models[column] = rf

# Evaluasi model Random Forest
for column in y.columns:
    print(f"Classification Report untuk {column} (Random Forest):")
    print(classification_report(y_test[column], y_pred_rf[column]))
    print(f"Accuracy untuk {column} (Random Forest): {accuracy_score(y_test[column], y_pred_rf[column])}\n")

# Simpan seluruh model TensorFlow dan Random Forest dalam satu file
joblib.dump(rf_models, 'rf_models_all_labels_V2.pkl')
```
## Results
To check the result, we will try to fill the random input into the model first.
```bash
# Memuat seluruh model dari file
models_loaded = joblib.load('rf_models_all_labels_V2.pkl')


# Misalnya, ingin melakukan prediksi untuk input baru
input_baru = pd.DataFrame({
    'Q1': [3], 'Q2': [2], 'Q3': [2], 'Q4': [2], 'Q5': [1],
    'Q6': [1], 'Q7': [1], 'Q8': [1], 'Q9': [1], 'Q10': [1],
    'Q11': [1], 'Q12': [1], 'Q13': [2], 'Q14': [2], 'Q15': [2],
    'Q16': [1], 'Q17': [1], 'Q18': [1], 'Q19': [1], 'Q20': [1],
    'Q21': [1], 'Q22': [1], 'Q23': [1], 'Q24': [1], 'Q25': [1]
})

# Melakukan prediksi untuk setiap gangguan
prediksi = {}
for gangguan in models_loaded.keys():
    model = models_loaded[gangguan]
    prediksi[gangguan] = model.predict(input_baru)[0]

# Menampilkan hasil prediksi untuk semua gangguan
for gangguan, hasil in prediksi.items():
    print(f"Prediksi {gangguan}: {hasil}")
```
*output :* 
*Prediksi Level Kecemasan: 2
Prediksi Level Depresi: 1
Prediksi Level Bipolar: 2
Prediksi Level Skizofrenia: 1
Prediksi Level OCD: 1*

## License
Copyright © 2024 Mently Group . All rights reserved.



