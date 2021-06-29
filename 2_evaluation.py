#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Evaluating the combined model script for category verification task.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2021
---------------------------
"""

import sys
from copy import deepcopy
from logging import getLogger, basicConfig, INFO
from pathlib import Path

from pandas import read_csv, DataFrame

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.data.category_verification_data import CategoryVerificationItemData, apply_substitution_if_available
from framework.evaluation.column_names import CLOCK, OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.utils import decompose_multiword

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# arg choices: dataset
ARG_DATASET_TRAIN = "train"
ARG_DATASET_TEST  = "test"

# Paths
root_input_dir = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")


def main(spec: CategoryVerificationJobSpec, decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue):

    cv_item_data: CategoryVerificationItemData = CategoryVerificationItemData()

    model_output_dir = Path(root_input_dir, spec.output_location_relative())

    for category_label, object_label in cv_item_data.category_object_pairs():
        model_output_path = Path(model_output_dir, f"{category_label}-{object_label}.csv")
        if not model_output_path.exists():
            logger.warning(f"{model_output_path.name} not found.")
            continue

        object_label_sensorimotor = apply_substitution_if_available(object_label, cv_item_data.substitutions_sensorimotor)
        object_label_linguistic = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
        object_label_linguistic_multiword_parts = decompose_multiword(object_label_linguistic)

        model_data: DataFrame = read_csv(model_output_path, header=0, index_col=CLOCK, dtype={CLOCK: int})

        # set level prior to SOA
        below_no = (
                (model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[spec.soa_ticks] < decision_threshold_no)
                or any(model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(oll)].loc[spec.soa_ticks] < decision_threshold_no
                       for oll in object_label_linguistic_multiword_parts))
        above_yes = (
                (model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[spec.soa_ticks] > decision_threshold_yes)
                or any(model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(oll)].loc[spec.soa_ticks] > decision_threshold_yes
                       for oll in object_label_linguistic_multiword_parts))

        decision_made = False
        for tick in range(spec.soa_ticks + 1, spec.run_for_ticks):
            previously_below_no = below_no
            previously_above_yes = above_yes
            below_no = (
                    (model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[tick] < decision_threshold_no)
                    or any(model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(oll)].loc[tick] < decision_threshold_no
                           for oll in object_label_linguistic_multiword_parts))
            above_yes = (
                    (model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[tick] > decision_threshold_yes)
                    or any(model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(oll)].loc[tick] > decision_threshold_yes
                           for oll in object_label_linguistic_multiword_parts))

            if below_no and not previously_below_no:
                logger.info(f"{category_label}-{object_label}: {tick} NO!")
                decision_made = True
                break
            if above_yes and not previously_above_yes:
                logger.info(f"{category_label}-{object_label}: {tick} YES!")
                decision_made = True
                break
        if not decision_made:
            logger.info(f"{category_label}-{object_label}: UNDECIDED!")
        pass




if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    loaded_specs = CategoryVerificationJobSpec.load_multiple(Path(Path(__file__).parent,
                                                                 "job_specifications",
                                                                 "2021-06-25 search for more sensible parameters.yaml"))
    systematic_cca_test = True

    if systematic_cca_test:
        ccas = [1.0, 0.5, 0.0]
        specs = []
        s: CategoryVerificationJobSpec
        for s in loaded_specs:
            for cca in ccas:
                spec = deepcopy(s)
                spec.cross_component_attenuation = cca
                specs.append(spec)
    else:
        specs = loaded_specs

    for spec in specs:
        main(spec=spec,
             decision_threshold_no=0.5,
             decision_threshold_yes=0.9)

    logger.info("Done!")

