#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Compute coverage of the validation set.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2022
---------------------------
"""

from pathlib import Path
from sys import argv

from pandas import DataFrame

from framework.cognitive_model.ldm.preferences.preferences import Preferences as LDMPreferences
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.utils.logging import print_progress
from framework.cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from framework.cognitive_model.utils.logging import logger
from framework.data.category_verification_data import CategoryVerificationItemDataBlockedValidation, \
    CategoryVerificationItemData
from framework.data.col_names import ColNames


def is_member(dataset: CategoryVerificationItemData, category_label: str, object_label: str) -> bool:
    """
    Returns True if the category/object pair is present in the dataset and is tagged as being a true member, and
    False if the category/object pair is present and marked as being a false member.

    Raises a KeyError if the category/object pair is not present in the dataset.
    """
    return dataset._get_col_value(category_label, object_label, col_name=ColNames.ShouldBeVerified)


def main(linguistic_words: int):

    cv_item_data = CategoryVerificationItemDataBlockedValidation()
    fd = FreqDist.load(LDMPreferences.source_corpus_metas.bbc.freq_dist_path)
    sm = SensorimotorNorms(use_breng_translation=True)

    validation_coverage_data = []
    i = 0
    for category_label, object_label in cv_item_data.category_object_pairs():
        i += 1
        print_progress(i, len(cv_item_data.category_object_pairs()))

        object_rank = fd.rank(object_label, top_is_zero=False)
        object_in_lc = object_rank is not None and object_rank <= linguistic_words
        object_in_smc = sm.has_word(object_label)

        validation_coverage_data.append({
            "Category": category_label,
            "Object": object_label,
            "Is member": is_member(cv_item_data, category_label, object_label),
            "Object freq": fd[object_label],
            "Object rank": object_rank,
            "Object in LC": object_in_lc,
            "Object in SMC": object_in_smc,
            "Object in both": object_in_lc and object_in_smc,
        })

    df = DataFrame(validation_coverage_data)

    with Path("/Users/caiwingfield/Desktop/validation_coverage.csv").open("w") as f:
        df.to_csv(f, index=False)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(argv))

    main(
        linguistic_words=40_000,
    )

    logger.info("Done!")
