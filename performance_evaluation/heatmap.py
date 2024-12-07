import os
import json
import sys
import enum
import matplotlib.pyplot as plt
import seaborn as sns
from classifiers.dbod import IndividualDistanceBasedKeystrokeFeatureOutlierDetector
from classifiers.template_generator import all_ids, read_compact_format
from features.keystroke_features import (
    create_kht_data_from_df,
    create_kit_data_from_df,
    word_hold,
)
from rich.progress import track
import classifiers.verifiers_library as vl
from features.word_parser import SentenceParser

path = os.path.dirname(os.getcwd())
print(path)
sys.path.insert(0, path)


class VerifierType(enum.Enum):
    """Enum class representing the different types of verifiers available."""

    RELATIVE = 1
    SIMILARITY = 2
    SIMILARITY_UNWEIGHTED = 3
    ABSOLUTE = 4
    ITAD = 5


def get_user_by_platform(user_id, platform_id, session_id=None):
    """
    Retrieve data for a given user and platform, with an optional session_id filter.

    Parameters:
    - user_id (int): Identifier for the user.
    - platform_id (int or list[int]): Identifier for the platform.
      If provided as a list, it should contain two integers specifying
      an inclusive range to search between.
    - session_id (int or list[int], optional): Identifier for the session.
      If provided as a list, it can either specify an inclusive range with
      two integers or provide multiple session IDs to filter by.

    Returns:
    - DataFrame: Filtered data matching the given criteria.

    Notes:
    - When providing a list for platform_id or session_id to specify a range,
      the order of the two integers does not matter.
    - When providing a list with more than two integers for session_id,
      it will filter by those exact session IDs.

    Raises:
    - AssertionError: If platform_id or session_id list does not follow the expected format.

    Examples:
    >>> df = get_user_by_platform(123, 1)
    >>> df = get_user_by_platform(123, [1, 5])
    >>> df = get_user_by_platform(123, 1, [2, 6])
    >>> df = get_user_by_platform(123, 1, [2, 3, 4])

    """
    # Get all of the data for a user amd platform with am optional session_id

    # print(f"user_id:{user_id}", end=" | ")
    df = read_compact_format()
    if session_id is None:
        if isinstance(platform_id, list):
            # Should only contain an inclusive range of the starting id and ending id
            assert len(platform_id) == 2
            if platform_id[0] < platform_id[1]:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[0], platform_id[1]))
                ]
            else:
                return df[
                    (df["user_ids"] == user_id)
                    & (df["platform_id"].between(platform_id[1], platform_id[0]))
                ]

        return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]
    if isinstance(session_id, list):
        # Should only contain an inclusive range of the starting id and ending id
        if len(session_id) == 2:
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].between(session_id[0], session_id[1]))
            ]
        elif len(session_id) > 2:
            test = df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]
            # print(session_id)
            # print(test["session_id"].unique())
            # input()
            return df[
                (df["user_ids"] == user_id)
                & (df["platform_id"] == platform_id)
                & (df["session_id"].isin(session_id))
            ]

    return df[
        (df["user_ids"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


class HeatMap:
    """
    A heatmap generates the representative matrices for KHT, KIT, optionally word level features, or all of them
    and making a heatmap plot out of it
    """

    def __init__(self, verifier_type, p1=10, p2=10):
        self.verifier_type = verifier_type  # The verifier class to be used
        self.p1_threshold = p1
        self.p2_threshold = p2
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            self.config = json.load(f)
        print(f"----selected {verifier_type}")

    def make_kht_matrix(
        self, enroll_platform_id, probe_platform_id, enroll_session_id, probe_session_id
    ):
        """
        Make a matrix of KHT features from the enrollment and probe id's and
        their respective session id's
        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """
        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")

        matrix = []
        # TODO: We have to do a better job of figuring out how many users there
        # are automatically so we don't need to keep changing it manually
        ids = all_ids()
        for i in track(ids):
            print(i)
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kht_data_from_df(df)
            row = []
            # TODO: We have to do a better job of figuring out how many users there
            # are automatically so we don't need to keep changing it manually
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kht_data_from_df(df)
                v = vl.Verify(enrollment, probe, self.p1_threshold, self.p2_threshold)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def make_kit_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        """
        Make a matrix of KIT features from the enrollment and probe id's,
        their respective session id's, and the KIT flight feature (1-4)

        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """

        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        print(self.verifier_type)
        matrix = []
        ids = all_ids()
        for i in track(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kit_data_from_df(df, kit_feature_type)
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kit_data_from_df(df, kit_feature_type)
                v = vl.Verify(enrollment, probe)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def combined_keystroke_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        """
        Make a combined matrix of KIT and KHT features

        Note if enroll_platform_id, probe_platform_id are None, then all ids are used.
        But if one of them are None the other must also be none
        """
        # if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        #     raise ValueError("Platform ID must be between 1 and 3")

        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        ids = all_ids()
        for i in track(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            print(
                f"enroll_platform_id: {enroll_platform_id}, enroll_session_id: {enroll_session_id}, df.shape: {df.shape}"
            )
            print(
                f"probe_platform_id: {probe_platform_id}, probe_session_id: {probe_session_id}, df.shape: {df.shape}"
            )
            kht_enrollment = create_kht_data_from_df(df)
            kit_enrollment = create_kit_data_from_df(df, kit_feature_type)
            if self.config["use_word_holder"]:
                sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
                word_list = sp.get_words(df)
                word_hold_enrollment = word_hold(word_list, df)
                combined_enrollment = (
                    kht_enrollment | kit_enrollment | word_hold_enrollment
                )
            else:
                combined_enrollment = kht_enrollment | kit_enrollment
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                kht_probe = create_kht_data_from_df(df)
                kit_probe = create_kit_data_from_df(df, kit_feature_type)
                # outlier_detector = (
                #     IndividualDistanceBasedKeystrokeFeatureOutlierDetector(
                #         set(kit_probe.keys()), kit_probe
                #     )
                # )
                # kit_probe = outlier_detector.find_kht_or_kit_specific_inliers()
                if self.config["use_word_holder"]:
                    word_list = sp.get_words(df)
                    word_hold_probe = word_hold(word_list, df)
                    combined_probe = kht_probe | kit_probe | word_hold_probe
                else:
                    combined_probe = kht_probe | kit_probe
                v = vl.Verify(combined_enrollment, combined_probe)
                if self.verifier_type == VerifierType.ABSOLUTE:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.SIMILARITY:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SIMILARITY_UNWEIGHTED:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.ITAD:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def plot_heatmap(self, matrix, title=None):
        """Generate a heatmap from the provided feature matrix and optional title"""
        ax = sns.heatmap(matrix, linewidth=0.5).set_title(title)
        plt.savefig(title)
