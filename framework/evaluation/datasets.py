from enum import Enum, auto


# noinspection PyArgumentList
# this is an IDE bug!
class ParticipantDataset(Enum):
    """Which participant dataset to use with ROC plotting."""
    # All or none
    # Initial experiment
    original = auto()  # Original participant set
    replication = auto()  # Replication participant set
    original_plus_replication = auto()
    # Validation experiment
    validation = auto()  # Validation participant set
    balanced = auto()  # Validation (balanced study) participant set
    validation_plus_balanced = auto()
