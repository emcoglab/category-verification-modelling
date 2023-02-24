from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

from pandas import DataFrame, read_csv

from framework.evaluation.column_names import CLOCK
from framework.data.category_verification_data import CategoryVerificationItemDataOriginal, CategoryObjectPair, Filter, \
    CategoryVerificationItemDataBlockedValidation, CategoryVerificationItemDataValidationBalanced, \
    CategoryVerificationItemDataReplication
from framework.evaluation.datasets import ParticipantDataset


_logger = getLogger(__file__)


def load_model_output_from_dir(activation_traces_dir: Path, use_assumed_object_label: bool, validation: bool, for_participant_dataset: ParticipantDataset, with_filter: Optional[Filter] = None) -> Dict[CategoryObjectPair, DataFrame]:
    """
    Returns a CategoryObjectPair-keyed dictionary of activation traces.
    """
    _logger.info(f"\tLoading model activation logs from {activation_traces_dir.as_posix()}")

    if validation:
        if for_participant_dataset == ParticipantDataset.validation:
            category_item_pairs = CategoryVerificationItemDataBlockedValidation().category_object_pairs(with_filter)
        elif for_participant_dataset == ParticipantDataset.balanced:
            category_item_pairs = CategoryVerificationItemDataValidationBalanced().category_object_pairs(with_filter)
        else:
            raise NotImplementedError()
    else:
        # It doesn't matter which we pick
        assert CategoryVerificationItemDataOriginal().category_object_pairs() == CategoryVerificationItemDataReplication().category_object_pairs()

        category_item_pairs = CategoryVerificationItemDataOriginal().category_object_pairs(with_filter, use_assumed_object_label=use_assumed_object_label)

    # (object, item) -> model_data
    all_model_data: Dict[CategoryObjectPair, DataFrame] = dict()
    for category_item_pair in category_item_pairs:
        category_label, object_label = category_item_pair
        model_output_path = Path(activation_traces_dir, f"{category_label}-{object_label} activation.csv")
        if not model_output_path.exists():
            # logger.warning(f"{model_output_path.name} not found.")
            continue

        all_model_data[CategoryObjectPair(category_label, object_label)] = read_csv(model_output_path,
                                                                                    header=0, index_col=CLOCK,
                                                                                    dtype={CLOCK: int})

    if len(all_model_data) == 0:
        raise FileNotFoundError(f"No model data in {activation_traces_dir.as_posix()}")

    return all_model_data
