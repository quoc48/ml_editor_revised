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

data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())


