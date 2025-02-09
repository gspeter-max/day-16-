import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate random text (fake product descriptions)
text_samples = [
    "High-quality wireless headphones with noise cancellation.",
    "Ergonomic gaming chair with lumbar support.",
    "Smartphone with AI-powered camera and fast charging.",
    "Ultra HD 4K smart TV with Dolby Vision.",
    "Portable Bluetooth speaker with deep bass.",
    "Mechanical keyboard with RGB backlighting.",
    "Smartwatch with fitness tracking and heart rate monitor.",
    "Gaming laptop with RTX 4060 GPU and 32GB RAM.",
    "Mirrorless camera with 24MP sensor and 4K video.",
    "Electric standing desk with memory height adjustment."
]
text_data = np.random.choice(text_samples, num_samples)

# Generate fake image paths (assuming you have an 'images/' directory)
image_paths = [f"images/product_{i}.jpg" for i in range(num_samples)]

# Generate numerical features (random 5D vectors)
num_features = [np.random.rand(5).tolist() for _ in range(num_samples)]

# Generate structured numerical data
price = np.round(np.random.uniform(50, 1000, num_samples), 2)  # Prices between $50 and $1000
rating = np.round(np.random.uniform(1.0, 5.0, num_samples), 1)  # Ratings between 1.0 and 5.0

# Generate binary labels (0 = low quality, 1 = high quality)
labels = np.random.choice([0, 1], num_samples)

# Create DataFrame
df = pd.DataFrame({
    'id': range(1, num_samples + 1),
    'text': text_data,
    'image_path': image_paths,
    'price': price,
    'rating': rating,
    'num_features': num_features,
    'label': labels
})
print("✅ Dataset generated successfully! Check 'multimodal_dataset.csv'.")


import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text  import TfidfVectorizer

tfidf = TfidfVectorizer(max_features = 1000)
text_vectors = tfidf.fit_transform(df['text']).toarray()


import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

reset_net = ResNet50(weights='imagenet',include_top = False, pooling = 'avg')

def get_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (224, 224))
    image = np.expand_dims(image,axis = 0 )
    features = reset_net.predict(image)
    return features.flatten()

image_path = '/content/download.jpg'
temp_features = [get_features(image_path)]
important_features = [get_features(img_path) for img_path in df['image_path']]

from sklearn.model_selection import train_test_split 

x = np.hstack((text_vectors, numerical_features))
y = df['label'].values 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

from sklearn.svm import SVC

svc = SVC(
    kernel  = 'rbf', 
    C = 0.1, 
    gamma = 'scale', 
    probability= True
)
svc.fit(x_train, y_train)

from sklearn.metrics import classification_report , roc_auc_score

y_pre = svc.predict(x_test)
print(f'classification report : {classification_report(y_test, y_pre)}')
print(f'roc_auc_score : {roc_auc_score(y_test,svc.predict_proba(x_test)[:,1])}')


import optuna 

def objective(trial):
    m = trial.suggest_float("C", 1e-3 , 1e3, log = True )
    gamma = trial.suggest_float("gamma",1e-4,1e0, log = True )

    svc = SVC(kernel = 'rbf', C = m, gamma = gamma, probability= True)
    svc.fit(x_train, y_train)

    return roc_auc_score(y_test, svc.predict_proba(x_test)[:,1])

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 20)

best_params = study.best_params 
print(f"best params : {best_params}")
import shap

# Train the SVM model
svc = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svc.fit(x_train, y_train)  # Make sure the model is trained!

# Create SHAP Explainer using the trained model's `predict`
explainer = shap.Explainer(svc.predict, x_train)

# Compute SHAP values
shap_values = explainer(x_test)

# Plot summary
shap.summary_plot(shap_values, x_test)
