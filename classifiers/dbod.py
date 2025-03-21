import os
import json
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class DistanceBasedKeystrokeFeatureOutlierDetector:
    def __init__(self, common_features, enrollment_pattern, probe_pattern) -> None:
        # Since our data was in nanoseconds, I changed the r from 100 to 1000000000 to define a bigger range
        # FIXME: we will probably need to adjust the self.r, 100 is definitely too small but 1e9 maybe too big we will have to see what performs the best
        #        100000000 seems to improve performance slightly
        with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
            config = json.load(f)
        self.r = int(config["dbod_r"])
        self.beta = float(config["dbod_beta"])
        self.common_features = common_features
        self.enrollment_pattern = enrollment_pattern
        self.probe_pattern = probe_pattern

    def find_inliers(self):
        inlier_enrollment_features = defaultdict(list)
        inlier_probe_features = defaultdict(list)
        for feature in self.common_features:
            timings = self.enrollment_pattern[feature]
            # print("Timings:", list(timings))
            if len(timings) == 1:
                inlier_enrollment_features[feature].append(timings[0])
                continue
            elif len(timings) == 2:
                inlier_enrollment_features[feature].append(timings[0])
                inlier_enrollment_features[feature].append(timings[1])
                continue
            elif len(timings) == 3:
                inlier_enrollment_features[feature].append(timings[0])
                inlier_enrollment_features[feature].append(timings[1])
                inlier_enrollment_features[feature].append(timings[2])
                continue

            # TODO: Maybe another thing that might reduce some of the empty lists is to just force include all timings where len(timings) = 2 as well? It really depends on how frequent it is
            for timing in timings:
                # Establish the neighborhood for the current timing, so that all other timings can compare against the neighborhood
                # Trying a wider range for the neighborhood by making the lower bound negative
                lower_neighborhood_bound = -(timing - self.r)
                upper_neighborhood_bound = timing + self.r
                # Filter the current timing (make a copy of the list to make sure it doesn't get mutated in-place)
                timings_to_compare_against = []
                for timing_candidate in timings:
                    if timing != timing_candidate:
                        timings_to_compare_against.append(timing_candidate)
                # Count how many of the remaining timings except the current fall in the neighborhood
                count = len(
                    list(
                        filter(
                            lambda x: lower_neighborhood_bound
                            <= x
                            <= upper_neighborhood_bound,
                            timings_to_compare_against,
                        )
                    )
                )
                # print("lower_neighborhood_bound", lower_neighborhood_bound)
                # print("upper_neighborhood_bound", upper_neighborhood_bound)
                # input()
                # print("Timings to compare:", list(timings_to_compare_against))
                # input()
                # print("neighborhood count:", count)
                # input()
                # See if counts/number of timings - 1 (because we removed 1 timing value) >= the beta param
                if (count / (len(timings) - 1)) >= self.beta:
                    inlier_enrollment_features[feature].append(timing)

        for feature in self.common_features:
            timings = self.probe_pattern[feature]
            if len(timings) == 1:
                inlier_probe_features[feature].append(timings[0])
                continue
            elif len(timings) == 2:
                inlier_probe_features[feature].append(timings[0])
                inlier_probe_features[feature].append(timings[1])
                continue
            elif len(timings) == 3:
                inlier_probe_features[feature].append(timings[0])
                inlier_probe_features[feature].append(timings[1])
                inlier_probe_features[feature].append(timings[2])
                continue

            for timing in timings:
                # Establish the neighborhood for the current timing, so that all other timings can compare against the neighborhood
                lower_neighborhood_bound = timing - self.r
                upper_neighborhood_bound = timing + self.r
                # Filter the current timing (make a copy of the list to make sure it doesn't get mutated in-place)
                timings_to_compare_against = filter(lambda x: x != timing, timings)
                # Count how many of the remaining timings except the current fall in the neighborhood
                count = len(
                    list(
                        filter(
                            lambda x: lower_neighborhood_bound
                            <= x
                            <= upper_neighborhood_bound,
                            timings_to_compare_against,
                        )
                    )
                )
                # See if counts/number of timings - 1 (because we removed 1 timing value) >= the beta param
                if (count / (len(timings) - 1)) >= self.beta:
                    inlier_probe_features[feature].append(timing)
        return (inlier_enrollment_features, inlier_probe_features)

    def find_inliers_with_lof(self, contamination=0.1, n_neighbors=20):
        """Detect inliers using Local Outlier Factor (LOF)."""
        inlier_enrollment_features = defaultdict(list)
        inlier_probe_features = defaultdict(list)
        for feature in self.common_features:
            timings = self.enrollment_pattern[feature]
            if len(timings) > 1:
                timings = np.array(timings).reshape(-1, 1)
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, contamination=contamination
                )
                inlier_mask = lof.fit_predict(timings) == 1
                inlier_enrollment_features[feature] = (
                    timings[inlier_mask].flatten().tolist()
                )
            else:
                inlier_enrollment_features[feature] = timings
        for feature in self.common_features:
            timings = self.probe_pattern[feature]
            if len(timings) > 1:
                timings = np.array(timings).reshape(-1, 1)
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, contamination=contamination
                )
                inlier_mask = lof.fit_predict(timings) == 1
                inlier_probe_features[feature] = timings[inlier_mask].flatten().tolist()
            else:
                inlier_probe_features[feature] = timings

        return (inlier_enrollment_features, inlier_probe_features)

    def find_inliers_with_isolation_forest(self, contamination=0.1):
        """Detect inliers using Isolation Forest."""
        inlier_enrollment_features = defaultdict(list)
        inlier_probe_features = defaultdict(list)
        for feature in self.common_features:
            timings = self.enrollment_pattern[feature]
            if len(timings) > 1:
                timings = np.array(timings).reshape(-1, 1)
                iso_forest = IsolationForest(
                    contamination=contamination, random_state=42
                )
                inlier_mask = iso_forest.fit_predict(timings) == 1
                inlier_enrollment_features[feature] = (
                    timings[inlier_mask].flatten().tolist()
                )
            else:
                inlier_enrollment_features[feature] = timings
        for feature in self.common_features:
            timings = self.probe_pattern[feature]
            if len(timings) > 1:
                timings = np.array(timings).reshape(-1, 1)
                iso_forest = IsolationForest(
                    contamination=contamination, random_state=42
                )
                inlier_mask = iso_forest.fit_predict(timings) == 1
                inlier_probe_features[feature] = timings[inlier_mask].flatten().tolist()
            else:
                inlier_probe_features[feature] = timings

        return (inlier_enrollment_features, inlier_probe_features)
