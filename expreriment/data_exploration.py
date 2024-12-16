import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import os

from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(Path('data/writers.csv'))

# Start by changing types to ake precessing easier
df["AnswerCount"] = df["AnswerCount"].fillna(-1)
df["AnswerCount"] = df["AnswerCount"].astype(int)
df["PostTypeId"] = df["PostTypeId"].astype(int)
df["Id"] = df["Id"].astype(int)
df.set_index("Id", inplace=True, drop=False)

# Add measure of the length of a post
df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
df["text_len"] = df["full_text"].str.len()

# A question is a post of id 1
df["is_question"] = df["PostTypeId"] == 1

df = df[df["PostTypeId"].isin([1,2])]


fig = plt.figure(figsize=(16,10))
fig.suptitle("Distribution of number of answers for high and low score questions")
plt.xlim(-5,80)

ax = df[df["is_question"] &
        (df["Score"] > df["Score"].median())]["AnswerCount"].hist(bins=60,
                                                          density=True,
                                                          histtype="step",
                                                          color="orange",
                                                          linewidth=3,
                                                          grid=False,
                                                          figsize=(16, 10))

df[df["is_question"] &
   ~(df["Score"] > df["Score"].median())]["AnswerCount"].hist(bins=60,
                                                     density=True,
                                                     histtype="step",
                                                     color="purple",
                                                     linewidth=3,
                                                     grid=False)

handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in
           ["orange", "purple"]]
labels = ["High score", "Low score"]
plt.legend(handles, labels)
ax.set_xlabel("Num answers")
ax.set_ylabel("Percentage of sentences");

# Show the plot
plt.show()