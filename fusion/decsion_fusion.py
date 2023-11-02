from performance_evaluation.heatmap import VerifierType


def verdict(similarity_score, verifier_type):
    """
    Determines the verdict based on the similarity score and verifier type.

    Parameters:
    - similarity_score (float): A similarity measure between two entities.
      The value should be within [0, 1] with 1 indicating perfect similarity and 0 indicating no similarity.
    - verifier_type (VerifierType): The type of verifier being used.
      Must be one of the values from the VerifierType enum.

    Returns:
    - bool: The verdict based on the provided similarity score and verifier type.
      Returns False if the similarity score exceeds the defined threshold for the specified verifier type,
      otherwise returns True.

    Notes:
    - ABSOLUTE and SIMILARITY verifiers have an empirically defined threshold of 0.8.
    - ITAD verifier has an empirically defined threshold of 0.25.
      The reason for this specific threshold is due to the coding approach of ITAD which has set `p` to 0 implicitly,
      causing perfect matches to cap around 0.25.

    Example:
    >>> verdict(0.9, VerifierType.ABSOLUTE)
    False
    >>> verdict(0.2, VerifierType.ITAD)
    True

    """
    # We have individual empirically defined verdict thresholds for each verifier
    # print(verifier_type)
    # print(similarity_score)
    if verifier_type == VerifierType.ABSOLUTE and similarity_score >= 0.8:
        return False
    elif verifier_type == VerifierType.SIMILARITY and similarity_score >= 0.8:
        return False
    # I think with how we coded ITAD our perfect matches cap just about 0.25 and the non-matches are much lower
    # most likely this is because we set p to 0 implicitly
    elif verifier_type == VerifierType.ITAD and similarity_score >= 0.25:
        return False
    else:
        return True


def is_fake_profile(verdicts):
    """
    Determines if a profile is fake based on the majority verdict.

    Parameters:
    - verdicts (list[bool]): A list of verdicts where each verdict indicates whether
      a profile is genuine (False) or fake (True).

    Returns:
    - str: A verdict on the overall profile authenticity.
      Returns "Genuine" if the number of genuine verdicts is greater than or equal to the number
      of fake verdicts, otherwise returns "Fake".

    Note:
    The function uses a simple majority rule to determine the profile's authenticity.
    It counts the number of genuine and fake verdicts and compares them to make a decision.
    """
    # The verdicts then will determine if a profile is fake or not
    # and we take a simple majority to see whether the fake counts beat out the
    # genuine counts
    genuine_count = verdicts.count(False)
    fake_profile_count = verdicts.count(True)
    if genuine_count >= fake_profile_count:
        return "Genuine"
    return "Fake"


def get_actual_designation(enrollment_id, probe_id):
    """
    Determines the actual designation of a profile based on the enrollment and probe IDs.

    Parameters:
    - enrollment_id (int or str): The ID corresponding to a profile's enrollment.
    - probe_id (int or str): The ID corresponding to a profile's probe.

    Returns:
    - str: The actual designation of the profile.
      Returns "Genuine" if the enrollment ID matches the probe ID, indicating that
      the profile is genuine. Otherwise, returns "Fake".

    Note:
    A genuine profile is defined as having matching enrollment and probe IDs.

    Example:
    >>> get_actual_designation(101, 101)
    'Genuine'
    >>> get_actual_designation(102, 103)
    'Fake'

    """
    # We define a genuine profile to have the same enrollment_id and probe_id
    if enrollment_id == probe_id:
        return "Genuine"
    return "Fake"
