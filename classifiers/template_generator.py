import os
import json
import pandas as pd
import numpy as np


class Genders:
    """
    A utility class that provides static methods for gender categories.

    This class offers static methods to retrieve standardized representations
    of different gender categories. The available categories are ALL, MALE, FEMALE, and OTHER.

    Usage:
    - Genders.ALL()   -> "all"
    - Genders.MALE()  -> "male"
    - Genders.FEMALE()-> "female"
    - Genders.OTHER() -> "other"
    """

    @staticmethod
    def ALL():
        """
        Returns the string representation for the category that includes all genders.

        Returns:
        - str: "all"
        """
        return "all"

    @staticmethod
    def MALE():
        """
        Returns the string representation for the male gender category.

        Returns:
        - str: "male"
        """
        return "male"

    @staticmethod
    def FEMALE():
        """
        Returns the string representation for the female gender category.

        Returns:
        - str: "female"
        """
        return "female"

    @staticmethod
    def OTHER():
        """
        Returns the string representation for the other gender category.

        Returns:
        - str: "other"
        """
        return "other"


def read_compact_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "dataset", "cleaned2.csv"),
        dtype={
            "key": str,
            "press_time": np.float64,
            "release_time": np.float64,
            "platform_id": np.uint8,
            "session_id": np.uint8,
            "user_ids": np.uint8,
        },
    )
    # print(df.head())
    return df


def all_ids():
    """
    Retrieve a list of IDs based on the gender type specified in the configuration file.

    This function reads a configuration file named 'classifier_config.json' located in the current
    working directory. It then extracts the gender type from the file and returns a corresponding list
    of IDs. The IDs are assigned based on predefined gender categories.

    Returns:
    - list[int]: A list of IDs corresponding to the specified gender type.

    Raises:
    - ValueError: If the gender type in the configuration file is unrecognized.

    Preconditions:
    - The 'classifier_config.json' file should be present in the current working directory.
    - The file should contain a valid JSON structure with a 'gender' field.
    """
    with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
        config = json.load(f)
    gender_type = str(config["gender"])
    if gender_type.lower() == Genders.ALL().lower():
        df = read_compact_format()
        return list(set(df["user_ids"].tolist()))
    elif gender_type.lower() == Genders.MALE().lower():
        return [9, 12, 14, 15, 16, 17, 18, 20, 21, 26, 27]
    elif gender_type.lower() == Genders.FEMALE().lower():
        return [1, 3, 4, 5, 6, 8, 10, 11, 13, 19, 22, 23, 24]
    elif gender_type.lower() == Genders.OTHER().lower():
        return [2, 7, 25]
    else:
        raise ValueError(f"Unknown gender type {gender_type}")
