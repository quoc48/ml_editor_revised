import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.externals import joblib
import random
from collections import defaultdict
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

from ml_editor.data_processing import (
    format_raw_df,
    get_split_by_author,
    add_text_features_to_df,
    get_vectorized_series, 
    get_feature_vector_and_label
)
from ml_editor.model_v1 import get_model_probabilities_for_input_texts
from lime.lime_text import LimeTextExplainer
random.seed(40)

# 1. Load the data
data_path = Path('/data/writers.csv')
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

#4 use LIME to make functions that will explain a model's predictions for a specific example
def explain_one_instance(instance, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(instance, get_model_probabilities_for_input_texts, num_features=6)
    return exp

def visualize_one_exp(features, labels, index, class_names = ["Low score","High score"]):
    exp = explain_one_instance(features[index], class_names = class_names)
    print('Index: %d' % index)
    print('True class: %s' % class_names[labels[index]])
    exp.show_in_notebook(text=True)

visualize_one_exp(list(test_df["full_text"]), list(y_test), 7)
