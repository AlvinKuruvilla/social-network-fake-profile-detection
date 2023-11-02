import os
import pandas as pd
import spacy
import numpy as np


# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    """
    A helper function that takes as input a dataframe, and returns a new dataframe
    no longer containing rows with the string "<0>".

    Parameters:
    - df: a pandas DataFrame.

    Returns:
    - DataFrame without rows containing the string "<0>".
    """
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]


def clean_letters(letters):
    """
    Removes single quotation marks from each item in a list of letters/strings.

    Parameters:
    - letters (list[str]): A list of strings or letters, each potentially wrapped with single quotation marks.

    Returns:
    - list[str]: A list of strings or letters without the surrounding single quotation marks.

    Note:
    This function is designed to clean up lists where each item might be wrapped with single quotation marks
    (e.g., ["'a'", "'b'", "'c'"] to ["a", "b", "c"]).

    Example:
    >>> clean_letters(["'a'", "'b'", "'hello'"])
    ['a', 'b', 'hello']

    """
    return [item.strip("'") for item in letters]


class SentenceParser:
    """Parses our dataset into best-effort sentences by trying to account for punctuation.

    We them use the spacy tokenizer to tokenize the best-effort sentences into words for word level features
    """

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def path(self):
        return self.csv_file_path

    def as_df(self):
        return pd.read_csv(
            os.path.join(os.getcwd(), "cleaned.csv"),
            dtype={
                "key": str,
                "press_time": np.float64,
                "release_time": np.float64,
                "platform_id": np.uint8,
                "session_id": np.uint8,
                "user_ids": np.uint8,
            },
        )

    def letters(self, as_list: bool = False):
        if as_list == True:
            return list(remove_invalid_keystrokes(self.as_df()).iloc[:, 1])
        elif as_list == False:
            return remove_invalid_keystrokes(self.as_df())

    def make_sentences(self, raw_df):
        ignorable = ["Key.cmd", "Key.tab", "Key.shift", "Key.shift_r"]
        filtered = []
        keys = clean_letters(remove_invalid_keystrokes(raw_df).loc[:, "key"])
        for i, key in enumerate(keys):
            filtered.append(key)

        filtered_list = []
        for i in range(len(filtered) - 1):
            if filtered[i] == "Key.backspace":
                if not filtered_list:
                    filtered_list.pop()
            else:
                filtered_list.append(filtered[i])
        filtered = filtered_list

        sentence = ""
        for i, key in enumerate(filtered):
            if i + 1 < len(filtered) and i - 1 >= 0:
                next_index = (i + 1) % len(filtered)
                if not key in ignorable:
                    if key == "Key.enter" or key == "." or key == "!" or key == "?":
                        sentence += "\n"
                        continue
                    elif key == "Key.space":
                        sentence += " "
                        continue
                    elif key == "Key.backspace" or key == "Key.ctrl":
                        continue
                    elif filtered[i - 1] == "Key.ctrl":
                        sentence += "\n"
                        continue
                    elif filtered[next_index] == "Key.backspace":
                        continue
                    sentence += key.strip()
        return sentence

    def get_words(self, data_df):
        tokenized_words = []
        sentences = self.make_sentences(data_df)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences)
        for token in doc:
            tokenized_words.append(token.text)
        return tokenized_words
