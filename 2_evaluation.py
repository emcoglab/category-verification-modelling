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

from __future__ import annotations

import sys
from copy import deepcopy
from enum import Enum
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Optional, List, Tuple

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


class Decision(Enum):
    Yes = 1
    No = 0


class _Decider:
    """
    Test a level of activation against an upper and lower threshold.
    Remembers the most recent test to inform the first time a decision is reached.
    """
    def __init__(self, threshold_yes: ActivationValue, threshold_no: ActivationValue):
        assert threshold_no < threshold_yes

        # Threshold for yes and no decisions
        self.threshold_yes: ActivationValue = threshold_yes
        self.threshold_no: ActivationValue = threshold_no

        # The current level of activation. None when no activation has historically been presented.
        self._last_activation: Optional[ActivationValue] = None

    @property
    def _above_yes(self) -> Optional[bool]:
        """True whenn activation is above the yes threshold (None for None)."""
        if self._last_activation is None:
            return None
        return self._last_activation >= self.threshold_yes

    @property
    def _below_no(self) -> Optional[bool]:
        """True whenn activation is below the no threshold (None for None)."""
        if self._last_activation is None:
            return None
        return self._last_activation <= self.threshold_no

    @property
    def _current_decision(self) -> Optional[Decision]:
        """The current decision, based on the current level of activation. None if no decision has been reached."""
        if self._above_yes:
            return Decision.Yes
        if self._below_no:
            return Decision.No
        return None

    def test_activation_level(self, activation: ActivationValue) -> Optional[Decision]:
        """Present a level of activation to test if it causes a decision to be reached."""
        previous_decision = self._current_decision
        self._last_activation = activation
        new_decision = self._current_decision
        if new_decision is not previous_decision:
            return new_decision
        else:
            return None

    @classmethod
    def combined_decision(cls, deciders: List[_Decider], activations: List[ActivationValue]) -> Optional[Tuple[Decision, int]]:
        """
        Present each activation to each of a list of deciders, and output the decision and index of the ordinally first
        one to make a decision.
        None if no decision.
        """
        assert len(deciders) > 0
        assert len(deciders) == len(activations)
        i: int
        d: _Decider
        for i, d in enumerate(deciders):
            decision = d.test_activation_level(activations[i])
            if decision is not None:
                return decision, i
        return None


def main(spec: CategoryVerificationJobSpec, decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue):

    cv_item_data: CategoryVerificationItemData = CategoryVerificationItemData()

    model_output_dir = Path(root_input_dir, spec.output_location_relative())

    for category_label, object_label in cv_item_data.category_object_pairs():
        model_output_path = Path(model_output_dir, f"{category_label}-{object_label}.csv")
        if not model_output_path.exists():
            logger.warning(f"{model_output_path.name} not found.")
            continue

        object_label_sensorimotor: str = apply_substitution_if_available(object_label, cv_item_data.substitutions_sensorimotor)
        object_label_linguistic: str = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
        object_label_linguistic_multiword_parts: List[str] = decompose_multiword(object_label_linguistic)

        model_data: DataFrame = read_csv(model_output_path, header=0, index_col=CLOCK, dtype={CLOCK: int})

        sensorimotor_decider = _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
        linguistic_deciders = [
            _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
            for _part in object_label_linguistic_multiword_parts
        ]

        decision_made = False
        for tick in range(spec.soa_ticks + 1, spec.run_for_ticks):
            sensorimotor_decision = sensorimotor_decider.test_activation_level(
                activation=model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[tick])
            linguistic_decision = _Decider.combined_decision(
                deciders=linguistic_deciders,
                activations=[
                    model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(part)].loc[tick]
                    for part in object_label_linguistic_multiword_parts
                ])

            if linguistic_decision is None and sensorimotor_decision is None:
                continue

            if linguistic_decision == Decision.No or sensorimotor_decision == Decision.No:
                logger.info(f"{category_label}-{object_label}: {tick} NO!")
                decision_made = True
                break
            if linguistic_decision == Decision.Yes or sensorimotor_decision == Decision.Yes:
                logger.info(f"{category_label}-{object_label}: {tick} YES!")
                decision_made = True
                break

        if not decision_made:
            logger.info(f"{category_label}-{object_label}: UNDECIDED!")


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
