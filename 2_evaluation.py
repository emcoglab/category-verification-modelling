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
from typing import List, Tuple, Dict, Optional

from matplotlib import pyplot
from numpy.random import seed
from scipy.stats import norm
from pandas import read_csv, DataFrame
from seaborn import heatmap

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
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
    # Above yes
    Yes = 1
    # Below no
    No = 0
    # In the indecisive region
    Undecided = -1
    # Hasn't yet entered the indecisive region
    Waiting = -2

    @property
    def made(self) -> bool:
        """Has a decision been positively made?"""
        if self == Decision.Yes:
            return True
        if self == Decision.No:
            return True
        return False


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

        self.last_decision: Decision = Decision.Waiting

    def _above_yes(self, activation) -> bool:
        """True whenn activation is above the yes threshold."""
        # In case the threshold is equal to the activation cap (i.e. FULL_ACTIVATION), floating-point arithmetic means
        # we may never reach it. Therefore in this instance alone we reduce the threshold minutely when testing for
        # aboveness.
        if self.threshold_yes == FULL_ACTIVATION:
            return activation >= self.threshold_yes - 1e-10
        else:
            return activation >= self.threshold_yes

    def _below_no(self, activation) -> bool:
        """True when activation is below the no threshold."""
        # In case the threshold is equal to zero (i.e. minimum activation), floating-point arithmetic means we may never
        # reach it. So we raise the threshold minutely in this case only when testing for belowness.
        if self.threshold_no == 0:
            return activation <= self.threshold_no + 1e-10
        else:
            return activation <= self.threshold_no

    def _make_decision(self, activation) -> Decision:
        """The current decision, based on the current level of activation."""
        # If we peak above YES in any scenario, that is a yes
        if self._above_yes(activation):
            return Decision.Yes
        # We can only decide "no" if we were previously undecided
        elif self.last_decision == Decision.Undecided:
            if self._below_no(activation):
                return Decision.No
            else:
                return Decision.Undecided
        elif self.last_decision == Decision.Waiting:
            if self._below_no(activation):
                return Decision.Waiting
            else:
                return Decision.Undecided
        else:
            raise RuntimeError()

    def test_activation_level(self, activation: ActivationValue) -> Decision:
        """Present a level of activation to test if it causes a decision to be reached."""
        decision = self._make_decision(activation)
        self.last_decision = decision
        return decision

    @classmethod
    def multi_tests(cls, deciders: List[_Decider], activations: List[ActivationValue]) -> List[Decision]:
        """
        Present each activation to each of a list of deciders, and output the decision from each.
        """
        assert 0 < len(deciders) == len(activations)
        decisions = []
        i: int
        d: _Decider
        for i, d in enumerate(deciders):
            decision = d.test_activation_level(activations[i])
            decisions.append(decision)
        return decisions


def make_model_decision(object_label, decision_threshold_no, decision_threshold_yes, cv_item_data, model_data, spec) -> Tuple[Decision, int]:
    object_label_sensorimotor: str = apply_substitution_if_available(object_label, cv_item_data.substitutions_sensorimotor)
    object_label_linguistic: str = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
    object_label_linguistic_multiword_parts: List[str] = decompose_multiword(object_label_linguistic)
    sensorimotor_decider = _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
    linguistic_deciders = [
        _Decider(threshold_yes=decision_threshold_yes, threshold_no=decision_threshold_no)
        for _part in object_label_linguistic_multiword_parts
    ]
    for tick in range(spec.soa_ticks + 1, spec.run_for_ticks):
        sensorimotor_decision = sensorimotor_decider.test_activation_level(
            activation=model_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].loc[tick])
        linguistic_decisions = _Decider.multi_tests(
            deciders=linguistic_deciders,
            activations=[
                model_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(part)].loc[tick]
                for part in object_label_linguistic_multiword_parts
            ])
        for decision in [sensorimotor_decision, *linguistic_decisions]:
            if decision.made:
                return decision, tick
    return Decision.Undecided, spec.run_for_ticks


def check_decision(decision: Decision, category_should_be_verified: bool) -> Tuple[bool, bool, bool, bool]:
    """Checks a Decision to see if it was correct. Returns True iff the decision is correct."""
    # When it's undecided or waiting, we default to no
    if (decision == Decision.Undecided) or (decision == Decision.Waiting):
        decision = Decision.No

    model_hit = category_should_be_verified and (decision == Decision.Yes)
    model_miss = category_should_be_verified and (decision == Decision.No)
    model_fa = (not category_should_be_verified) and (decision == Decision.Yes)
    model_cr = (not category_should_be_verified) and (decision == Decision.No)

    return model_hit, model_fa, model_cr, model_miss


def correct_rate_for_thresholds(all_model_data: Dict[Tuple[str, str], DataFrame],
                                decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue,
                                spec: CategoryVerificationJobSpec, save_dir: Path) -> Tuple[float, float]:
    """Returns correct_rate and dprime."""

    zed = norm.ppf

    ground_truth_dataframe = CV_ITEM_DATA.dataframe

    model_guesses = []
    category_item_pairs: List[Tuple[str, str]] = CV_ITEM_DATA.category_object_pairs()
    for category_label, object_label in category_item_pairs:
        item_is_of_category: bool = CV_ITEM_DATA.is_correct(category_label, object_label)

        model_decision: Decision
        decision_made_at_time: int
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

        model_hit, model_fa, model_cr, model_miss = check_decision(model_decision, item_is_of_category)

        model_guesses.append((
            category_label, object_label,
            item_is_of_category,
            model_decision, decision_made_at_time,
            model_hit, model_fa, model_cr, model_miss,
            model_hit or model_cr,
        ))
    model_guesses = DataFrame.from_records(model_guesses, columns=[
        ColNames.CategoryLabel, ColNames.ImageObject,
        ColNames.ShouldBeVerified,
        "Model decision", "Decision made at time",
        "Model HIT", "Model FA", "Model CR", "Model MISS",
        "Model is correct",
    ])

    # Rates computed as proportion of all trials, not all trials on which the model can decide
    n_trials = len(category_item_pairs)
    model_correct_rate = len(model_guesses[model_guesses["Model is correct"]]) / n_trials
    model_hit_rate = len(model_guesses[model_guesses["Model HIT"]]) / len(model_guesses[model_guesses[ColNames.ShouldBeVerified] == True])
    model_false_alarm_rate = len(model_guesses[model_guesses["Model FA"]]) / len(model_guesses[model_guesses[ColNames.ShouldBeVerified] == False])

    # This is a simple Y/N task, not a 2AFC, so we can just use standard d-prime
    if (model_hit_rate == 0) or (model_false_alarm_rate == 1):
        # No hits, or all false alarms. dprime should be -inf, but let's call it -10 so we can see it
        model_dprime = -10
    elif (model_false_alarm_rate == 0) or (model_hit_rate == 1):
        # No false alarms, or all hits, dprime should be info, but let's call it +10 so we can see it
        model_dprime = 10
    else:
        model_dprime = zed(model_hit_rate) - zed(model_false_alarm_rate)

    results_dataframe = ground_truth_dataframe.merge(model_guesses,
                                                     how="left", on=[ColNames.CategoryLabel, ColNames.ImageObject])

    # Save individual threshold data for verification
    save_dir.mkdir(parents=True, exist_ok=True)
    with Path(save_dir, f"no{decision_threshold_no}_yes{decision_threshold_yes}.csv").open("w") as f:
        results_dataframe.to_csv(f, header=True, index=False)

    return model_correct_rate, model_dprime


def save_heatmap(hitrates: DataFrame, path: Path, value_col: str, vlims: Tuple[Optional[float], Optional[float]]):
    vmin, vmax = vlims
    pivot = hitrates.pivot(index="Decision threshold (no)", columns="Decision threshold (yes)", values=value_col)
    ax = heatmap(pivot,
                 annot=True, fmt=".02f",
                 vmin=vmin, vmax=vmax, cmap='Reds',
                 square = True,
                 )
    ax.figure.savefig(path)
    pyplot.close(ax.figure)


def main(spec: CategoryVerificationJobSpec):

    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    save_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative(), " evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Only load the model data once, then just reference it for each hitrate.
    # TODO: This is turning into spaghetti code, but let's get it working first.
    logger.info(f"\tLoading model activation logs from {model_output_dir.as_posix()}")
    # (object, item) -> model_data
    all_model_data: Dict[Tuple[str, str], DataFrame] = dict()
    for category_label, object_label in CV_ITEM_DATA.category_object_pairs():
        model_output_path = Path(model_output_dir, "activation traces", f"{category_label}-{object_label} activation.csv")
        if not model_output_path.exists():
            # logger.warning(f"{model_output_path.name} not found.")
            continue

        all_model_data[(category_label, object_label)] = read_csv(model_output_path, header=0, index_col=CLOCK, dtype={CLOCK: int})

    if len(all_model_data) == 0:
        logger.warning(f"No model data in {model_output_dir.as_posix()}")
        return

    hitrates = []
    dprimes = []
    for decision_threshold_no in THRESHOLDS:
        for decision_threshold_yes in THRESHOLDS:
            if decision_threshold_no >= decision_threshold_yes:
                continue
            hitrate, dprime = correct_rate_for_thresholds(all_model_data=all_model_data,
                                                          decision_threshold_yes=decision_threshold_yes,
                                                          decision_threshold_no=decision_threshold_no,
                                                          spec=spec, save_dir=Path(save_dir, "hitrates by threshold"))
            hitrates.append((decision_threshold_no, decision_threshold_yes, hitrate))
            dprimes.append((decision_threshold_no, decision_threshold_yes, dprime))

    # Save overall dprimes
    dprimes_df = DataFrame.from_records(
        dprimes,
        columns=["Decision threshold (no)", "Decision threshold (yes)", "d-prime"])
    with Path(save_dir, "dprimes overall.csv").open("w") as f:
        dprimes_df.to_csv(f, header=True, index=False)
    save_heatmap(dprimes_df, Path(save_dir, "dprimes overall.png"), value_col="d-prime", vlims=(None, None))

    logger.info(f"Largest hitrate this model: {max(t[2] for t in hitrates)}")
    logger.info(f"Largest dprime this model: {max(t[2] for t in dprimes if -10<t[2]<10)}")


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    loaded_specs = CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-06-25 search for more sensible parameters.yaml"))
    loaded_specs.extend(CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-07-15 40k different decay.yaml")))
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
