from collections import defaultdict


def create_kht_data_from_df(df):
    """
    Computes Key Hold Time (KHT) data from a given dataframe.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    Returns:
    - dict: A dictionary where keys are individual key characters and values are lists containing
      computed KHT values (difference between the release time and press time) for each instance of the key.

    Note:
    KHT is defined as the difference between the release time and the press time for a given key instance.
    This function computes the KHT for each key in the dataframe and aggregates the results by key.
    """
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type):
    """
    Computes Key Interval Time (KIT) data from a given dataframe based on a specified feature type.

    Parameters:
    - df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    - kit_feature_type (int): Specifies the type of KIT feature to compute. The valid values are:
      1: Time between release of the first key and press of the second key.
      2: Time between release of the first key and release of the second key.
      3: Time between press of the first key and press of the second key.
      4: Time between press of the first key and release of the second key.

    Returns:
    - dict: A dictionary where keys are pairs of consecutive key characters and values are lists containing
      computed KIT values based on the specified feature type for each instance of the key pair.

    Note:
    This function computes the KIT for each pair of consecutive keys in the dataframe and aggregates
    the results by key pair. The method for computing the KIT is determined by the `kit_feature_type` parameter.
    """
    kit_dict = defaultdict(list)
    if df.empty:
        # print("dig deeper: dataframe is empty!")
        return kit_dict
    num_rows = len(df.index)
    for i in range(num_rows):
        if i < num_rows - 1:
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            key = current_row["key"] + next_row["key"]
            initial_press = float(current_row["press_time"])
            second_press = float(next_row["press_time"])
            initial_release = float(current_row["release_time"])
            second_release = float(next_row["release_time"])
            if kit_feature_type == 1:
                kit_dict[key].append(second_press - initial_release)
            elif kit_feature_type == 2:
                kit_dict[key].append(second_release - initial_release)
            elif kit_feature_type == 3:
                kit_dict[key].append(second_press - initial_press)
            elif kit_feature_type == 4:
                kit_dict[key].append(second_release - initial_press)
    return kit_dict


def word_hold(word_list, raw_df):
    """
    Computes best-effort Word Hold Time (WH) for each word in a given word list from a raw dataframe of keystrokes.
    A word's WH is defined as the time difference between the release of its last character and the press
    of its first character. Non-printing keys (e.g., Shift) are currently not handled.

    Parameters:
    - word_list (list of str): A list of words for which the WH needs to be computed.
      The words are generated from the SentenceParser's make_words function.

    - raw_df (pandas.DataFrame): A dataframe with columns "key", "press_time", and "release_time",
      where each row represents an instance of a key press and its associated press and release times.

    Returns:
    - dict: A dictionary where keys are words from the word_list and values are lists containing
      computed WH values (difference between the release time of the last key and the press time
      of the first key) for each instance of the word.
    """
    wh = defaultdict(list)
    # The word_list needs to be in the same order as they would
    # sequentially appear in the dataframe
    raw_df["visited"] = False
    for word in word_list:
        first_letter = word[0]
        # print(first_letter)
        potential_release_matches = raw_df[
            (~raw_df["visited"]) & (raw_df["key"].str.strip("'") == first_letter)
        ]
        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            # print(first_row_index)
            # print(raw_df.loc[first_row_index])
            # input()
            # TODO: How to account for shift and other non-printing keys that could appear in between?
            # The ending bound is exclusive
            press_time = raw_df.loc[first_row_index]["press_time"]
            try:
                release_time = raw_df.loc[first_row_index + len(word) - 1][
                    "release_time"
                ]
                raw_df.loc[
                    first_row_index : first_row_index + len(word) - 1, "visited"
                ] = True
            except KeyError:
                release_time = raw_df.loc[first_row_index + len(word) - 2][
                    "release_time"
                ]
                raw_df.loc[
                    first_row_index : first_row_index + len(word) - 2, "visited"
                ] = True
            wh[word].append(release_time - press_time)
    return wh
