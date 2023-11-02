import os
import statistics
import json
from classifiers.ecdf import ECDF


class Verify:
    """
    Our implementations of the Similarity (both weighted and unweighted),
    Absolute, Relative, and ITAD verifiers
    """

    def __init__(self, p1, p2, p1_t=10, p2_t=10):
        # p1 and p2 are dictionaries of features
        # keys in the dictionaries would be the feature names
        # feature names mean individual letters for KHT
        # feature names could also mean pair of letters for KIT or diagraphs
        # feature names could also mean pair of sequence of three letters for trigraphs
        # feature names can be extended to any features that we can extract from keystrokes
        self.pattern1 = p1
        self.pattern2 = p2
        self.pattern1threshold = (
            p1_t  # sort of feature selection, based on the availability
        )
        self.pattern2threshold = (
            p2_t  # sort of feature selection, based on the availability
        )
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            config = json.load(f)
        self.common_features = []
        if config["use_feature_selection"]:
            for feature in self.pattern1.keys():
                if feature in self.pattern2.keys():
                    if (
                        len(self.pattern1[feature]) >= self.pattern1threshold
                        and len(self.pattern2[feature]) >= self.pattern2threshold
                    ):
                        self.common_features.append(feature)
        else:
            self.common_features = set(self.pattern1.keys()).intersection(
                set(self.pattern2.keys())
            )
        # print(f"comparing {len(self.common_features)} common_features")

    def get_abs_match_score(self):  # A verifier
        """
        Computes the absolute matching score between two patterns based on their common features.

        The method checks the ratio of medians of each common feature in both patterns. If the ratio is
        below a threshold (currently set to 1.5), it considers the feature as a match. The final score
        is the proportion of matched features to the total common features.

        The function assumes that the class instance has the attributes:
        - self.common_features: a list of features that are common between two patterns.
        - self.pattern1: a dictionary where keys are feature names and values are lists of values for pattern 1.
        - self.pattern2: a dictionary where keys are feature names and values are lists of values for pattern 2.

        Returns:
        - float: The absolute matching score which is a ratio of matched features to total common features.

        Raises:
        - ValueError: If there are no common features or if an unexpected zero median is encountered.

        Notes:
        If there are no common features or minimum of medians of a feature is 0, the function currently returns a score of 0.
        """
        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # TODO: When running the performance_evaluation with cleaned2.csv, this ValueError gets proced
            raise ValueError("Error: no common features to compare!")
        matches = 0
        for (
            feature
        ) in self.common_features:  # checking for every common feature for match
            # print(f"feature:{feature}")
            # print(f"self.pattern1[feature]:{self.pattern1[feature]}")
            # print(f"self.pattern2[feature]:{self.pattern2[feature]}")

            pattern1_median = statistics.median(self.pattern1[feature])
            pattern2_median = statistics.median(self.pattern2[feature])
            if min(pattern1_median, pattern2_median) == 0:
                return 0  # Must look into and fix this! just a temporary arrangment
                # raise ValueError('min of means is zero, should not happen!')
            else:
                ratio = max(pattern1_median, pattern2_median) / min(
                    pattern1_median, pattern2_median
                )
            # the following threshold is what we thought would be good
            threshold = 1.5
            if ratio <= threshold:  # basically the current feature matches
                matches += 1
        return matches / len(self.common_features)

    def get_similarity_score(self):  # S verifier, each key same weight
        """
        Computes the similarity score between two patterns based on their common features.

        The similarity score is calculated by first computing the median and standard deviation (stdev)
        of the time values for each common feature in pattern 1. For each time value in pattern 2 for the same
        feature, the function checks if the value lies within one standard deviation from the median of pattern 1.

        A feature is considered a match if more than half of its time values in pattern 2 lie within this range.
        The final similarity score is the ratio of matched features to the total common features.

        The function assumes that the class instance has the attributes:
        - self.common_features: a list of features that are common between two patterns.
        - self.pattern1: a dictionary where keys are feature names and values are lists of values for pattern 1.
        - self.pattern2: a dictionary where keys are feature names and values are lists of values for pattern 2.

        Returns:
        - float: The similarity score which is a ratio of matched features to total common features.

        Notes:
        If there are no common features, the function returns a score of 0. In the case where the standard deviation
        cannot be computed (e.g., when a feature has only one value), the function defaults to using a quarter of
        the median value as the stdev.
        """

        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        key_matches, total_features = 0, 0
        for feature in self.common_features:
            pattern1_median = statistics.median(list(self.pattern1[feature]))
            try:
                pattern1_stdev = statistics.stdev(self.pattern1[feature])
            except statistics.StatisticsError:
                # print("In error: ", self.pattern1[feature])
                if len(self.pattern1[feature]) == 1:
                    pattern1_stdev = self.pattern1[feature][0] / 4
                else:
                    pattern1_stdev = (
                        self.pattern1[feature] / 4
                    )  # this will always be one value that is when exception would occur

            value_matches, total_values = 0, 0
            for time in self.pattern2[feature]:
                if (pattern1_median - pattern1_stdev) < time and time < (
                    pattern1_median + pattern1_stdev
                ):
                    value_matches += 1
                total_values += 1
            if value_matches / total_values <= 0.5:
                key_matches += 1
            total_features += 1

        return key_matches / total_features

    def get_weighted_similarity_score(
        self,
    ):  # S verifier, each feature different weights
        """
        Computes the weighted similarity score between two patterns based on their common features.

        The weighted similarity score is calculated using the following steps:
        1. Compute the median (as a proxy for the mean) and standard deviation (stdev) of the time values for
           each common feature in pattern 1 (referred to as the "enrollment" pattern).
        2. For each time value in pattern 2 (referred to as the "template" pattern) for the same feature,
           check if the value lies within one standard deviation from the median of the enrollment pattern.
        3. Sum the number of matches and total values for all features.
        4. Compute the ratio of total matches to total values for all features to get the final similarity score.

        The function assumes that the class instance has the attributes:
        - self.common_features: a list of features that are common between two patterns.
        - self.pattern1: a dictionary where keys are feature names and values are lists of values for pattern 1.
        - self.pattern2: a dictionary where keys are feature names and values are lists of values for pattern 2.

        Returns:
        - float: The weighted similarity score which is a ratio of total matched values to total values across all
                 features.

        Notes:
        If there are no common features, the function returns a score of 0. In cases where the standard deviation
        cannot be computed (e.g., when a feature has only one value), the function defaults to using a quarter of
        the median value as the stdev.

        """

        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        matches, total = 0, 0
        for feature in self.common_features:
            enroll_mean = statistics.median(list(self.pattern1[feature]))
            try:
                template_stdev = statistics.stdev(self.pattern1[feature])
            except statistics.StatisticsError:
                # print("In error: ", self.pattern1[feature])
                if len(self.pattern1[feature]) == 1:
                    template_stdev = self.pattern1[feature][0] / 4
                else:
                    template_stdev = self.pattern1[feature] / 4

            for time in self.pattern2[feature]:
                if (enroll_mean - template_stdev) < time and time < (
                    enroll_mean + template_stdev
                ):
                    matches += 1
                total += 1
        return matches / total

    def get_cdf_xi(self, distribution, sample):
        """
        Computes the cumulative distribution function (CDF) value at a given sample
        point based on the provided distribution.

        Parameters:
        - distribution (list or array-like): The list of data points representing the distribution.
        - sample (float or int): The point at which to evaluate the CDF.

        Returns:
        - float: The CDF value of the given sample in the provided distribution.
        """
        ecdf = ECDF(distribution)
        prob = ecdf(sample)
        # print('prob:', prob)
        return prob

    def itad_similarity(self):  # The new one
        """
        Computes the ITAD similarity score
        between two typing patterns based on their shared features.

        The score represents the similarity between two patterns based on the cumulative
        distribution function (CDF) of the median values of the shared features.
        If a value from pattern2 is less than or equal to the median of the corresponding
        feature in pattern1, the CDF value at that point is used. Otherwise, 1 minus the CDF
        value is used.

        Returns:
        - float: The ITAD similarity score, which is the average of the computed similarities
          for all shared features.

        Preconditions:
        - It assumes the existence of a `get_cdf_xi` method to compute the CDF value at a given
          sample point.
        """

        # https://www.scitepress.org/Papers/2023/116841/116841.pdf
        if len(self.common_features) == 0:  # this wont happen at all, but just in case
            # print("dig deeper: there is no common feature to match!")
            return 0
        similarities = []
        for feature in self.common_features:
            M_g_i = statistics.median(self.pattern1[feature])
            for x_i in self.pattern2[feature]:
                if x_i <= M_g_i:
                    similarities.append(self.get_cdf_xi(self.pattern1[feature], x_i))
                else:
                    similarities.append(
                        1 - self.get_cdf_xi(self.pattern1[feature], x_i)
                    )
        return statistics.mean(similarities)

    def scaled_manhattan_distance(self):
        """
        Computes the Scaled Manhattan Distance between two typing patterns based on their shared features.

        This metric calculates the distance by taking the absolute difference between
        a value from one pattern and the mean of the corresponding feature in the other pattern.
        This difference is then scaled by dividing it with the standard deviation of the
        corresponding feature. The computed distances for all shared features are then averaged
        to provide a final score.

        The Scaled Manhattan Distance gives an insight into how different the two patterns are
        in terms of their common features while accounting for the variability (standard deviation)
        of the features.

        Returns:
        - float: The averaged scaled manhattan distance for all shared features.

        """
        if (
            len(self.common_features) == 0
        ):  # this needs to be checked further when and why and for which users or cases it might hapens at all
            # print("dig deeper: there is no common feature to match!")
            return 0
        grand_sum = 0
        number_of_instances_compared = 0
        for feature in self.common_features:
            # print('comparing the feature:', feature)
            mu_g = statistics.mean(self.pattern1[feature])
            std_g = statistics.stdev(self.pattern1[feature])
            # print(f'mu_g:{mu_g}, and std_g:{std_g}')
            for x_i in self.pattern2[feature]:
                # print('x_i:', x_i)
                current_dist = abs(mu_g - x_i) / std_g
                # print('current_dist:', current_dist)
                grand_sum = grand_sum + current_dist
                # print('grand_sum:', grand_sum)
                number_of_instances_compared = number_of_instances_compared + 1
        # print('number_of_instances_compared', number_of_instances_compared)
        return grand_sum / number_of_instances_compared
