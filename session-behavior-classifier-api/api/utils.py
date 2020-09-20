import numpy as np


def deleteKeysFromDict(dictToRemove, keysToRemove):
    """
    Given a list of keys and a dictionary, remove 
    all the given keys from it if they exist.
    """
    for k in keysToRemove:
        try:
            del dictToRemove[k]
        except KeyError:
            pass
    for v in dictToRemove.values():
        if isinstance(v, dict):
            deleteKeysFromDict(v, keysToRemove)

    return dictToRemove


class transformData(object):
    """
    Class to handle loading and processing of raw datasets.
    """

    def __init__(self, input_data,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 input_size=400, num_of_classes=5):
        """
        Initialization of a Data object.

        Args:
            input_data ([str]): List of input strings
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.data = input_data

    def get_all_data(self):
        """
        Input data preprocessing

        Returns:
            (np.ndarray) Data transformed from raw to indexed form

        """
        return np.asarray([self.str_to_indexes(s) for s in self.data], dtype='int64')

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.

        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
