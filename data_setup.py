import os
import re
import numpy as np
import pandas as pd

from skmultilearn.model_selection import iterative_train_test_split

from typing import List, Tuple


def preprocessing(string: str) -> str:
    string = re.sub('\n', ' ', string)
    string = re.sub(r'[ㄱ-ㅎㅏ-ㅣ ]', ' ', string)
    string = re.sub(r'[^0-9가-힣,.!? ]', '', string)
    string = re.sub(r' *[.!?]+ *', '.', string)
    string = re.sub('\.([0-9]+)\.', '', string)
    string = re.sub(r'^[. ]+', '', string)
    string = re.sub(' +', ' ', string)
    return string

def get_data(data_path: str, file_name: str) -> pd.DataFrame:
    """
    Args:
        data_path (str): directory which csv file lives.
        file_name (str): file name with extension to load.

    Returns:
        df (pandas.DataFrame): columns with 'review'(str),
            and target (15 one-hot) 
    """
    df = pd.read_csv(os.path.join(data_path, file_name))
    df = pd.concat([df['review'], df.loc[:, df.dtypes==int]], axis=1)
    df['review'] = df['review'].map(preprocessing)
    return df

def data_split(dataframe: pd.DataFrame,
               test_size: float = 0.3,
               random_state: int = None) -> Tuple[np.array]:
    """
    Use `iterative_train_test_split` in scikit-multilearn library
    to split dataframe into training set and test set
    with feature(text) and target(15 one-hot)
    because of extremely imbalanced data.

    Args: 
        dataframe (pandas.DataFrame): dataframe trying to split.
        random_state (int)
        
    Returns:
        X_train (numpy.array): 
            Train text data array with index (index, text).
        y_train (numpy.array): Train target data (15 one-hot).
        X_test (numpy.array):
            Test text data array with index (index, text).
        y_test (numpy.array): Test target data (15 one-hot).
    """
    np.random.seed(random_state)
    X_array = np.array(dataframe[['review']].reset_index())
    y_array = np.array(dataframe.loc[:, dataframe.dtypes==int])

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X_array, y_array, test_size=test_size
    )
    return X_train, y_train, X_test, y_test

def long_form(dataframe: pd.DataFrame,
              train_index: List) -> pd.DataFrame:
    """
    Extract train data from train_index.
    And convert train dataframe to long form
    for train SBERT, which takes two sentence as input.

    Args:
        dataframe (pandas.DataFrame): 
            Dataframe for extract training set
        train_index (List): Training set index

    Returns:
        dataframe_melt (pandas.DataFrame):
            Long format of training dataset.
            columns = 'index' (same index is same review)
                      'review' (sentence 1), 
                      'variable' (sentence 2),
                      'value' (target, 0 or 1)
    """
    dataframe_melt = dataframe.loc[train_index, :]
    dataframe_melt = pd.melt(dataframe_melt.reset_index(),
                             id_vars=['index', 'review'])
    return dataframe_melt

def save_encoded(data_path: str,
                 file_name: str,
                 save_name: str,
                 random_state: int,
                 s_bert):
    df = get_data(data_path, file_name)
    X_train, y_train, X_test, y_test = data_split(dataframe=df,
                                       random_state=random_state)
    X_train_vectorized = s_bert.encode(X_train[:, 1].tolist())
    X_test_vectorized = s_bert.encode(X_test[:, 1].tolist())
    
    X_train_vectorized, y_train, X_test_vectorized, y_test

    np.savez(f"{data_path}/{save_name}",
        X_train_vectorized=X_train_vectorized,
        y_train=y_train,
        X_test_vectorized=X_test_vectorized,
        y_test=y_test)
