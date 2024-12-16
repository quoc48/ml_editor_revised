import pandas as pd
import spacy
import umap
import numpy as np
from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

# Add the project root (ML_EDITOR) to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import format_raw_df, get_random_train_test_split, get_vectorized_inputs_and_label, get_split_by_author

data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())

#1
train_df_rand, test_df_rand = get_random_train_test_split(df[df["is_question"]], test_size=0.3, random_state=40)

print("%s questions in training, %s in test." % (len(train_df_rand),len(test_df_rand)))
train_owners = set(train_df_rand['OwnerUserId'].values)
test_owners = set(test_df_rand['OwnerUserId'].values)
print("%s different owners in the training set" % len(train_df_rand))
print("%s different owners in the testing set" % len(test_df_rand))
print("%s owners appear in both sets" % len(train_owners.intersection(test_owners)))

#2
train_author, test_author = get_split_by_author(df[df["is_question"]], test_size=0.3, random_state=40)

print("%s questions in training, %s in test." % (len(train_author),len(test_author)))
train_owners = set(train_author['OwnerUserId'].values)
test_owners = set(test_author['OwnerUserId'].values)
print("%s different owners in the training set" % len(train_owners))
print("%s different owners in the testing set" % len(test_owners))
print("%s owners appear in both sets" % len(train_owners.intersection(test_owners)))