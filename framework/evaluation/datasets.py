from enum import Enum, auto


# noinspection PyArgumentList
# this is an IDE bug!
class ParticipantDataset(Enum):
    """Which participant dataset to use with ROC plotting."""
    # All or none
    all = auto()  # Both participant sets for either study
    # Initial experiment
    original = auto()  # Original participant set
    replication = auto()  # Replication participant set
    # Validation experiment
    validation = auto()  # Validation participant set
    balanced = auto()  # Validation (balanced study) participant set
