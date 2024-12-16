import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.sparse import vstack, hstack

def format_raw_df(df):
    """
    Cleanup data and join questions to answers
    :param df: raw DataFrame
    :return: processed DataFrame
    """
    # Fixing types and setting index
    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["AnswerCount"] = df["AnswerCount"].fillna(-1)
    df["AnswerCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    # Filtering out PostTypeIds other than documented ones
    df = df[df["PostTypeId"].isin([1, 2])]

    # Linking questions and answers
    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df

def get_vertorized_series(text_series, vectorizer):
    """
    Vectorizes an input series using a pre-trained vertorizer
    :param text_series: pandas Series of text
    :param vectorizer: pretrained sklearn vectorizer
    :return: array if vectorized feature
    """
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series


def get_normalized_series(df, col):
    """
    Get a normalized version of a column
    :param df: DataFrame
    :param col: column name
    :return: normalized series using z-score
    """
    return (df[col] - df[col].mean()) / df[col].std()

def get_vectorized_inputs_and_label(df):
    """
    Concatenate DataFrame features with text vectors
    :param df: DataFrame with calculated features
    :return: concatenated vector consisting of features and text
    """
    vectorized_features = np.append(
        np.vstack(df["vectors"]),
        df[
            [
                "action_verb_full",
                "question_mark_full",
                "norm_text_len",
                "language_question",
            ]
        ],
        1,
    )
    label = df["Score"] > df["Score"].median()
    return vectorized_features, label

def get_random_train_test_split(posts, test_size=0.3, random_state=40):
    """
    Get train/test split from DataFrame
    Assumes the DataFrame has one row per question example
    :param posts: all posts, with their labels
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    return train_test_split(
        posts, test_size=test_size, random_state=random_state
    )

def get_split_by_author(
    posts, author_id_column="OwnerUserId", test_size=0.3, random_state=40
):
    """
    Get train/test split
    Guarantee every author only appears in one of the splits
    :param posts: all posts, with their labels
    :param author_id_column: name of the column containing the author_id
    :param test_size: the proportion to allocate to test
    :param random_state: a random seed
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(posts, groups=posts[author_id_column])
    train_idx, test_idx = next(splits)
    return posts.iloc[train_idx, :], posts.iloc[test_idx, :]