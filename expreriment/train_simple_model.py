import pandas as pd
import numpy as np
import matplotlib as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import sys
sys.path.append("..")
np.random.seed(35)
import warnings
warnings.filterwarnings('ignore')

# Add the project root (ML_EDITOR) to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import (
    format_raw_df,
    add_text_features_to_df,
    get_split_by_author,
    get_vectorized_inputs_and_label,
    train_vectorizer,
    get_vectorized_series,
)

from ml_editor.model_v1 import get_model_probabilities_for_input_texts


data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())

df = df.loc[df["is_question"]].copy()

df = add_text_features_to_df(df.copy())
train_df, test_df = get_split_by_author(df, test_size=0.2, random_state=40)

vectorizer = train_vectorizer(train_df)
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

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', oob_score=True)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)
y_predicted_proba = clf.predict_proba(X_test)

y_train.value_counts()

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

# Training accuracy
y_train_pred = np.argmax(clf.oob_decision_function_,axis=1)

accuracy, precision, recall, f1 = get_metrics(y_train, y_train_pred)
print("Training accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
print("Validation accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

# Save the trained model and vectorizer to disk for futher analysis and use.
model_path = Path("../models/model_1.pkl")
vectorizer_path = Path("../models/vectorizer_1.pkl")
joblib.dump(clf, model_path) 
joblib.dump(vectorizer, vectorizer_path) 

# The inference function expects an array of questions, so we created an array of length 1 to pass a single question
test_q = ["bad question"]
probs = get_model_probabilities_for_input_texts(test_q)

# Index 1 corresponds to the positive class here
print("%s probability of the question receiving a high score according to our model" % (probs[0][1]))

