from logging import getLogger
from pathlib import Path
from typing import Dict

from pandas import DataFrame, read_csv

from framework.evaluation.column_names import CLOCK
from framework.data.category_verification_data import CategoryVerificationItemData, CategoryObjectPair
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize

_logger = getLogger(__file__)


def load_model_output_from_dir(model_output_dir: Path, exclude_repeated_items: bool) -> Dict[CategoryObjectPair, DataFrame]:
    """
    Returns a CategoryObjectPair-keyed dictionary of activation traces.
    """
    _logger.info(f"\tLoading model activation logs from {model_output_dir.as_posix()}")

    # (object, item) -> model_data
    all_model_data: Dict[CategoryObjectPair, DataFrame] = dict()
    for category_item_pair in CategoryVerificationItemData().category_object_pairs():
        if exclude_repeated_items and category_item_pair.is_repeated_item(using_tokenizer=modified_word_tokenize):
            continue
        category_label, object_label = category_item_pair
        model_output_path = Path(model_output_dir, "activation traces",
                                 f"{category_label}-{object_label} activation.csv")
        if not model_output_path.exists():
            # logger.warning(f"{model_output_path.name} not found.")
            continue

        all_model_data[CategoryObjectPair(category_label, object_label)] = read_csv(model_output_path,
                                                                                    header=0, index_col=CLOCK,
                                                                                    dtype={CLOCK: int})

    if len(all_model_data) == 0:
        raise FileNotFoundError(f"No model data in {model_output_dir.as_posix()}")

    return all_model_data
