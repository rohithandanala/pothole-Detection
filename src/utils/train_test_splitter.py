import random
from typing import List, Tuple

def train_test_split(
    image_list: List[str],
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits a list of images into training and testing sets.

    Args:
        image_list (List[str]): List of image filenames or paths.
        test_size (float, optional): Fraction of data to be used as test set. Defaults to 0.2.
        random_seed (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[List[str], List[str]]: (train_list, test_list)
    """
    random.seed(random_seed)
    image_list_copy = image_list.copy()
    random.shuffle(image_list_copy)

    split_idx = int(len(image_list_copy) * (1 - test_size))
    train_list = image_list_copy[:split_idx]
    test_list = image_list_copy[split_idx:]

    return train_list, test_list
