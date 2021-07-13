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
from numpy import linspace

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


def make_model_decision(object_label, decision_threshold_no, decision_threshold_yes, cv_item_data, model_data, spec) -> Optional[Decision]:
    object_label_sensorimotor: str = apply_substitution_if_available(object_label,
                                                                     cv_item_data.substitutions_sensorimotor)
    object_label_linguistic: str = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
    object_label_linguistic_multiword_parts: List[str] = decompose_multiword(object_label_linguistic)
    sensorimotor_decider = _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
    linguistic_deciders = [
        _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
        for _part in object_label_linguistic_multiword_parts
    ]
    decision_made = False
    decision = None
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
            decision_made = True
            decision = Decision.No
            break
        if linguistic_decision == Decision.Yes or sensorimotor_decision == Decision.Yes:
            decision_made = True
            decision = Decision.Yes
            break
    if not decision_made:
        decision = None
    return decision


def check_decision(decision: Optional[Decision], category_verification_correct: bool) -> bool:
    """Checks a Decision to see if it was correct. Returns True iff the deciion is correct."""
    # When it's undecided, we default to no
    if decision is None:
        decision = Decision.No
    if category_verification_correct:
        return decision == Decision.Yes
    else:
        return decision == Decision.No


def hitrate_for_thresholds(decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue,
                           spec: CategoryVerificationJobSpec):

    cv_item_data: CategoryVerificationItemData = CategoryVerificationItemData()

    model_output_dir = Path(root_input_dir, spec.output_location_relative())

    logger.info(f"Loading model activation logs from {model_output_dir.as_posix()}")

    model_correct_count: int = 0
    model_total_count: int = 0
    for category_label, object_label in cv_item_data.category_object_pairs():
        model_total_count += 1
        model_output_path = Path(model_output_dir, f"{category_label}-{object_label}.csv")
        if not model_output_path.exists():
            logger.warning(f"{model_output_path.name} not found.")
            continue

        model_data: DataFrame = read_csv(model_output_path, header=0, index_col=CLOCK, dtype={CLOCK: int})

        category_verification_correct: bool = cv_item_data.is_correct(category_label, object_label)

        decision: Optional[Decision] = make_model_decision(object_label,
                                                           decision_threshold_no, decision_threshold_yes,
                                                           cv_item_data, model_data,
                                                           spec)

        model_correct: bool = check_decision(decision, category_verification_correct)
        if model_correct:
            model_correct_count += 1

    model_hitrate = model_correct_count / model_total_count
    return model_hitrate


def main(spec: CategoryVerificationJobSpec):

    hitrates = []
    for decision_threshold_no in linspace(0, 1, 11):
        for decision_threshold_yes in linspace(0, 1, 11):
            if decision_threshold_no >= decision_threshold_yes:
                continue
            hitrate = hitrate_for_thresholds(decision_threshold_yes=decision_threshold_yes,
                                             decision_threshold_no=decision_threshold_no,
                                             spec=spec)
            hitrates.append((decision_threshold_no, decision_threshold_yes, hitrate))

    model_output_dir = Path(root_input_dir, spec.output_location_relative())
    with Path(model_output_dir, " hitrates.csv").open("w") as f:
        DataFrame.from_records(hitrates,
                               columns=["Decision threshold (no)", "Decision threshold (yes)", "hitrate"],
                               ).to_csv(f, header=True, index=False)


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
        main(spec=spec)

    logger.info("Done!")
