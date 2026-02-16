# ==============================
# IMPORTS
# ==============================

import os
import sys
import pandas as pd
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix project path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.preprocess import clean_text

# ==============================
# LOAD DATA
# ==============================

data_path = os.path.join(ROOT_DIR, "data", "resume_dataset.csv")
df = pd.read_csv(data_path)

print(df.columns)

df['Category'] = df['Category'].str.strip().str.lower()

# Use ONLY Resume_str (important)
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# ==============================
# TF-IDF (Stable Config)
# ==============================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=20000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    norm='l2'
)

X = vectorizer.fit_transform(df['cleaned_resume'])
y = df['Category']

# ==============================
# LABEL ENCODING
# ==============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==============================
# SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================

param_grid = {
    'C': [0.5, 1, 2, 5, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    LinearSVC(dual=False, max_iter=5000),
    param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# SAVE MODELS
# ==============================

model_dir = os.path.join(ROOT_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

print("\nModels saved successfully.")
