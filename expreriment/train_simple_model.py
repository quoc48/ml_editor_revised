import pandas as pd
import numpy as np
import matplotlib as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.externals import joblib
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
    get_feature_vector_and_label,
    get_split_by_author,
    get_vectorized_inputs_and_label,
    get_vectorized_series,
    train_vectorizer,
)

