import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.externals import joblib

import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import (
    format_raw_df, get_split_by_author, 
    add_text_features_to_df, 
    get_vectorized_series, 
    get_feature_vector_and_label
)

from ml_editor.model_evaluation import get_top_k

# 1. Load the data
data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())

# 2. Add features and split the dataset
df = add_text_features_to_df(df.loc[df["is_question"]].copy())
train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)

# 3. Load the trained model and vectorize the features
model_path = Path("../models/model_1.pkl")
clf = joblib.load(model_path) 
vectorizer_path = Path("../models/vectorizer_1.pkl")
vectorizer = joblib.load(vectorizer_path)

train_df["vectors"] = get_vectorized_series(train_df["full_text"].copy(), vectorizer)
test_df["vectors"] = get_vectorized_series(test_df["full_text"].copy(), vectorizer)
features = [
                "action_verb_full",
                "question_mark_full",
                "text_len",
                "language_question",
            ]

X_train, y_train = get_feature_vector_and_label(train_df, features)
X_test, y_test = get_feature_vector_and_label(test_df, features)

# Now, we'll use the top k method to look at:
# - The k best performing examples for each class (high and low scores)
# - The k worst performing examples for each class
# - The k most unsure examples, where our models prediction probability is close to .5
test_analysis_df = test_df.copy()
y_predicted_proba = clf.predict_proba(X_test)
test_analysis_df["predicted_proba"] = y_predicted_proba[:, 1]
test_analysis_df["true_label"] = y_test

to_display = [
    "predicted_proba",
    "true_label",
    "Title",
    "body_text",
    "text_len",
    "action_verb_full",
    "question_mark_full",
    "language_question",
]
threshold = 0.5

top_pos, top_neg, worst_pos, worst_neg, unsure = get_top_k(test_analysis_df, "predicted_proba", "true_label", k=2)
pd.options.display.max_colwidth = 500
top_pos[to_display]
top_neg[to_display]
worst_pos[to_display]
worst_neg[to_display]
unsure[to_display]



