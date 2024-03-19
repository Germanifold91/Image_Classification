from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import pandas as pd


def split_data(images_metadata: pd.DataFrame, split_params: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/catalog.yml.
    Returns:
        Split data.
    """
    random_state = split_params['random_state']
    test_size_training = split_params["test_size_training"]
    test_size_validation = split_params["test_size_validation"]

    X_train, y_train = train_test_split(
        images_metadata,test_size= test_size_training, random_state=random_state, stratify= images_metadata['labels']
        )
    X_test, y_test = train_test_split(
        y_train,test_size= test_size_validation, random_state=random_state, stratify= y_train['labels']
        )
    return X_train, X_test, y_train, y_test