import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.externals import joblib
from lime.lime_tabular import LimeTabularExplainer
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

from ml_editor.data_processing import (
    format_raw_df, 
    get_split_by_author,
    get_vectorized_series, 
    get_feature_vector_and_label
)
from ml_editor.model_evaluation import get_feature_importance
from ml_editor.model_v3 import get_question_score_from_input, get_features_from_input_text
from ml_editor.explanation_generation import FEATURE_DISPLAY_NAMES
from ml_editor.model_v2 import POS_NAMES

data_path = Path('../data/writers_with_features.csv')
df = pd.read_csv(data_path)

clf = joblib.load(Path("../models/model_3.pkl")) 
train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)

features = ["num_questions", 
               "num_periods",
               "num_commas",
               "num_exclam",
               "num_quotes",
               "num_colon",
               "num_stops",
               "num_semicolon",
               "num_words",
               "num_chars",
               "num_diff_words",
               "avg_word_len",
               "polarity"
              ]


y_test = test_df["Score"] > test_df["Score"].median()
X_test = test_df[features].astype(float)

features_and_labels = df.copy()
features_and_labels["label"] = features_and_labels["Score"] > features_and_labels["Score"].median()
features_and_labels = features_and_labels[["label"] + features].copy()

class_feature_values = features_and_labels.groupby("label").mean()
class_feature_values = class_feature_values.round(3)
class_feature_values = class_feature_values.transpose()
class_feature_values.rename(columns={
    False: "low score",
    True: "high score"
}, inplace=True)
class_feature_values["relative difference"] = abs(class_feature_values["high score"] - class_feature_values["low score"]) / class_feature_values["high score"]
class_feature_values.sort_values(by="relative difference", inplace=True, ascending=False)
class_feature_values

# Feature importance
k = 5

print("Using %s features in total" % len(features))
print("Top %s importances:\n" % k)
print('\n'.join(["%s: %.2g" % (tup[0], tup[1]) for tup in get_feature_importance(clf, np.array(features))[:k]]))

print("\nBottom %s importances:\n" % k)
print('\n'.join(["%s: %.2g" % (tup[0], tup[1]) for tup in get_feature_importance(clf, np.array(features))[-k:]]))

example_question = """
Is displaying a binary class enough to guide a user
"""
q_score = get_question_score_from_input(example_question)

print("%s probability of being a good question" % q_score)