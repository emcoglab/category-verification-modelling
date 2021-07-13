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
from typing import Optional, List, Tuple, Dict

from pandas import read_csv, DataFrame

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.data.category_verification_data import CategoryVerificationItemData, apply_substitution_if_available, \
    ColNames
from framework.evaluation.column_names import CLOCK, OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.utils import decompose_multiword

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# arg choices: dataset
ARG_DATASET_TRAIN = "train"
ARG_DATASET_TEST  = "test"

# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")

# Shared
CV_ITEM_DATA: CategoryVerificationItemData = CategoryVerificationItemData()
THRESHOLDS = [0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]  # linspace was causing weird float rounding errors


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


def make_model_decision(object_label, decision_threshold_no, decision_threshold_yes, cv_item_data, model_data, spec) -> Tuple[Optional[Decision], Optional[int]]:
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
    time = None
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
            time = tick
            break
        if linguistic_decision == Decision.Yes or sensorimotor_decision == Decision.Yes:
            decision_made = True
            decision = Decision.Yes
            time = tick
            break
    if not decision_made:
        decision = None
        time = None
    return decision, time


def check_decision(decision: Optional[Decision], category_verification_correct: bool) -> bool:
    """Checks a Decision to see if it was correct. Returns True iff the deciion is correct."""
    # When it's undecided, we default to no
    if decision is None:
        decision = Decision.No
    if category_verification_correct:
        return decision == Decision.Yes
    else:
        return decision == Decision.No


def hitrate_for_thresholds(all_model_data: Dict[Tuple[str, str], DataFrame],
                           decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue,
                           spec: CategoryVerificationJobSpec, save_dir: Path):

    ground_truth_dataframe = CV_ITEM_DATA.dataframe

    model_correct_count: int = 0
    model_total_count: int = 0
    model_guesses = []
    for category_label, object_label in CV_ITEM_DATA.category_object_pairs():
        model_total_count += 1

        category_verification_correct: bool = CV_ITEM_DATA.is_correct(category_label, object_label)

        model_decision: Optional[Decision]
        decision_made_at_time: Optional[int]
        try:
            model_data = all_model_data[(category_label, object_label)]
        # No model output was saved
        except KeyError:
            continue
        model_decision, decision_made_at_time = make_model_decision(object_label,
                                                                    decision_threshold_no, decision_threshold_yes,
                                                                    CV_ITEM_DATA,
                                                                    model_data,
                                                                    spec)

        model_correct: bool = check_decision(model_decision, category_verification_correct)

        model_guesses.append((category_label, object_label, model_decision, decision_made_at_time, model_correct))
        if model_correct:
            model_correct_count += 1
    model_guesses = DataFrame.from_records(model_guesses, columns=[
        ColNames.CategoryLabel, ColNames.ImageObject, "Model decision", "Decision made at time", "Model is correct"
    ])
    model_hitrate = model_correct_count / model_total_count

    results_dataframe = ground_truth_dataframe.merge(model_guesses,
                                                     how="left", on=[ColNames.CategoryLabel, ColNames.ImageObject])

    # Save individual threshold data for verification
    save_dir.mkdir(parents=True, exist_ok=True)
    with Path(save_dir, f"no{decision_threshold_no}_yes{decision_threshold_yes}.csv").open("w") as f:
        results_dataframe.to_csv(f, header=True, index=False)

    return model_hitrate


def main(spec: CategoryVerificationJobSpec):

    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    save_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative(), " evaluation", "hitrates")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Only load the model data once, then just reference it for each hitrate.
    # TODO: This is turning into spaghetti code, but let's get it working first.
    logger.info(f"Loading model activation logs from {model_output_dir.as_posix()}")
    # (object, item) -> model_data
    all_model_data: Dict[Tuple[str, str], DataFrame] = dict()
    for category_label, object_label in CV_ITEM_DATA.category_object_pairs():
        model_output_path = Path(model_output_dir, f"{category_label}-{object_label}.csv")
        if not model_output_path.exists():
            # logger.warning(f"{model_output_path.name} not found.")
            continue

        all_model_data[(category_label, object_label)] = read_csv(model_output_path, header=0, index_col=CLOCK, dtype={CLOCK: int})

    hitrates = []
    for decision_threshold_no in THRESHOLDS:
        for decision_threshold_yes in THRESHOLDS:
            if decision_threshold_no >= decision_threshold_yes:
                continue
            hitrate = hitrate_for_thresholds(all_model_data=all_model_data,
                                             decision_threshold_yes=decision_threshold_yes,
                                             decision_threshold_no=decision_threshold_no,
                                             spec=spec, save_dir=save_dir)
            hitrates.append((decision_threshold_no, decision_threshold_yes, hitrate))

    # Save overall hitrates
    with Path(save_dir, "overall.csv").open("w") as f:
        DataFrame.from_records(hitrates,
                               columns=["Decision threshold (no)", "Decision threshold (yes)", "hitrate"],
                               ).to_csv(f, header=True, index=False)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    loaded_specs = CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-06-25 search for more sensible parameters.yaml"))
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

    for i1, spec in enumerate(specs, start=1):
        logger.info(f"Evaluating model {i1} of {len(specs)}")
        main(spec=spec)

    logger.info("Done!")
