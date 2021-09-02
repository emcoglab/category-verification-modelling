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
from enum import Enum, auto
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from matplotlib import pyplot
from numpy import nan
from numpy.random import seed
from scipy.stats import norm
from pandas import read_csv, DataFrame
from seaborn import heatmap

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.utils.logging import print_progress
from framework.cognitive_model.version import VERSION
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


class Outcome(Enum):
    Hit = auto()
    Miss = auto()
    FalseAlarm = auto()
    CorrectRejection = auto()

    @property
    def name(self) -> str:
        if self == self.Hit:
            return "HIT"
        if self == self.Miss:
            return "MISS"
        if self == self.FalseAlarm:
            return "FA"
        if self == self.CorrectRejection:
            return "CR"

    @property
    def is_correct(self) -> bool:
        return self in {Outcome.Hit, Outcome.CorrectRejection}

    @property
    def answered_yes(self) -> bool:
        return self in {Outcome.Hit, Outcome.FalseAlarm}

    @classmethod
    def from_yn(cls, decide_yes: bool, should_be_yes: bool) -> Outcome:

        if should_be_yes:
            if decide_yes:
                return Outcome.Hit
            else:
                return Outcome.Miss
        else:
            if decide_yes:
                return Outcome.FalseAlarm
            else:
                return Outcome.CorrectRejection

    @classmethod
    def from_decision(cls, decision: Decision, should_be_yes: bool) -> Outcome:
        # When it's undecided or waiting, we default to no
        if (decision == Decision.Undecided) or (decision == Decision.Waiting):
            decision = Decision.No

        return cls.from_yn(decide_yes=decision == Decision.Yes, should_be_yes=should_be_yes)

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


def make_model_decision(object_label, decision_threshold_no, decision_threshold_yes, model_data, spec) -> Tuple[Decision, int]:
    object_label_sensorimotor: str = apply_substitution_if_available(object_label, CV_ITEM_DATA.substitutions_sensorimotor)
    object_label_linguistic: str = apply_substitution_if_available(object_label, CV_ITEM_DATA.substitutions_linguistic)
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


def performance_for_thresholds(all_model_data: Dict[Tuple[str, str], DataFrame],
                               exclude_repeated_items: bool,
                               decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue,
                               spec: CategoryVerificationJobSpec, save_dir: Path) -> Tuple[float, float, float]:
    """Returns correct_rate and dprime and criterion."""

    zed = norm.ppf

    ground_truth_dataframe = CV_ITEM_DATA.dataframe

    model_guesses = []
    category_item_pairs: List[Tuple[str, str]] = CV_ITEM_DATA.category_object_pairs()
    for category_label, object_label in category_item_pairs:
        item_is_of_category: bool = CV_ITEM_DATA.is_correct(category_label, object_label)

        try:
            model_data = all_model_data[(category_label, object_label)]
        # No model output was saved
        except KeyError:
            continue
        if exclude_repeated_items and is_repeated_item(category_label, object_label):
            continue

        model_decision: Decision
        decision_made_at_time: int
        model_decision, decision_made_at_time = make_model_decision(object_label,
                                                                    decision_threshold_no, decision_threshold_yes,
                                                                    model_data,
                                                                    spec)

        model_outcome: Outcome = Outcome.from_decision(decision=model_decision, should_be_yes=item_is_of_category)

        model_guesses.append((
            category_label, object_label,
            item_is_of_category,
            model_decision, decision_made_at_time,
            model_outcome.name, model_outcome.is_correct,
        ))
    model_guesses_df: DataFrame = DataFrame.from_records(model_guesses, columns=[
        ColNames.CategoryLabel, ColNames.ImageObject,
        ColNames.ShouldBeVerified,
        "Model decision", "Decision made at time",
        "Model outcome", "Model is correct",
    ])

    # Rates computed as proportion of all trials, not all trials on which the model can decide
    n_trials = len(category_item_pairs)
    model_correct_rate = len(model_guesses_df[model_guesses_df["Model is correct"]]) / n_trials
    model_hit_rate = len(model_guesses_df[model_guesses_df["Model outcome"] == Outcome.Hit.name]) / len(model_guesses_df[model_guesses_df[ColNames.ShouldBeVerified] == True])
    model_false_alarm_rate = len(model_guesses_df[model_guesses_df["Model outcome"] == Outcome.FalseAlarm.name]) / len(model_guesses_df[model_guesses_df[ColNames.ShouldBeVerified] == False])

    # This is a simple Y/N task, not a 2AFC, so we can just use standard d-prime
    if (model_hit_rate == 0) or (model_false_alarm_rate == 1) or (model_false_alarm_rate == 0) or (model_hit_rate == 1):
        # Can't compute a dprime or a criterion, so mark as missing
        model_dprime = nan
        model_criterion = nan
    else:
        model_dprime = zed(model_hit_rate) - zed(model_false_alarm_rate)
        model_criterion = - (zed(model_hit_rate) + zed(model_false_alarm_rate)) / 2

    results_dataframe = ground_truth_dataframe.merge(model_guesses_df,
                                                     how="left", on=[ColNames.CategoryLabel, ColNames.ImageObject])

    # Format columns
    results_dataframe["Decision made at time"] = results_dataframe["Decision made at time"].astype('Int64')

    # Save individual threshold data for verification
    save_dir.mkdir(parents=False, exist_ok=True)
    with Path(save_dir, f"no{decision_threshold_no}_yes{decision_threshold_yes}.csv").open("w") as f:
        results_dataframe.to_csv(f, header=True, index=False)

    return model_correct_rate, model_dprime, model_criterion


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


def is_repeated_item(category_label: str, object_label: str) -> bool:

    # Use the same decomposition/translation logic as elsewhere
    # TODO: this should be refactored into one place!
    category_label_linguistic: str = apply_substitution_if_available(category_label, CV_ITEM_DATA.substitutions_linguistic)
    category_label_sensorimotor: str = apply_substitution_if_available(category_label, CV_ITEM_DATA.substitutions_sensorimotor)
    object_label_linguistic: str = apply_substitution_if_available(object_label, CV_ITEM_DATA.substitutions_linguistic)
    object_label_sensorimotor: str = apply_substitution_if_available(object_label, CV_ITEM_DATA.substitutions_sensorimotor)
    category_label_linguistic_multiword_parts: List[str] = decompose_multiword(category_label_linguistic)
    object_label_linguistic_multiword_parts: List[str] = decompose_multiword(object_label_linguistic)

    all_category_words = set(category_label_linguistic_multiword_parts) | {category_label_sensorimotor}
    all_object_words = set(object_label_linguistic_multiword_parts) | {object_label_sensorimotor}

    # Repeated item if there's any word in common between category and object in either component
    return len(all_category_words.intersection(all_object_words)) > 0


def main(spec: CategoryVerificationJobSpec, exclude_repeated_items: bool, overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    if not model_output_dir.exists():
        logger.warning(f"Model out put not found for v{VERSION} in directory {model_output_dir.as_posix()}")
        return
    if not Path(model_output_dir, " MODEL RUN COMPLETE").exists():
        logger.info(f"Incomplete model run found in {model_output_dir.as_posix()}")
        return
    save_dir = Path(model_output_dir, " evaluation")
    if save_dir.exists() and not overwrite:
        logger.info(f"Evaluation complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=False, exist_ok=True)

    # Only load the model data once, then just reference it for each hitrate.
    # TODO: This is turning into spaghetti code, but let's get it working first.
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    criteria = []
    threshold_i = 0
    for decision_threshold_no in THRESHOLDS:
        for decision_threshold_yes in THRESHOLDS:
            if decision_threshold_no >= decision_threshold_yes:
                continue
            threshold_i += 1

            hitrate, dprime, criterion = performance_for_thresholds(
                all_model_data=all_model_data,
                exclude_repeated_items=exclude_repeated_items,
                decision_threshold_yes=decision_threshold_yes,
                decision_threshold_no=decision_threshold_no,
                spec=spec, save_dir=Path(save_dir, "hitrates by threshold"))
            hitrates.append((decision_threshold_no, decision_threshold_yes, hitrate))
            dprimes.append((decision_threshold_no, decision_threshold_yes, dprime))
            criteria.append((decision_threshold_no, decision_threshold_yes, criterion))

            print_progress(threshold_i, len(THRESHOLDS) * (len(THRESHOLDS) - 1) / 2, prefix="Running Yes/No thresholds: ", bar_length=50)

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'

    # Save overall dprimes
    dprimes_df = DataFrame.from_records(
        dprimes,
        columns=["Decision threshold (no)", "Decision threshold (yes)", "d-prime"])
    with Path(save_dir, f"{filename_prefix} dprimes.csv").open("w") as f:
        dprimes_df.to_csv(f, header=True, index=False)
    save_heatmap(dprimes_df, Path(save_dir, f"{filename_prefix} dprimes.png"), value_col="d-prime", vlims=(None, None))

    # Save overall criteria
    criteria_df = DataFrame.from_records(
        criteria,
        columns=["Decision threshold (no)", "Decision threshold (yes)", "criteria"])
    with Path(save_dir, f"{filename_prefix} criteria.csv").open("w") as f:
        criteria_df.to_csv(f, header=True, index=False)
    save_heatmap(criteria_df, Path(save_dir, f"{filename_prefix} criteria.png"), value_col="criteria", vlims=(None, None))

    logger.info(f"Largest hitrate this model: {max(t[2] for t in hitrates)}")
    logger.info(f"Largest dprime this model: {max(t[2] for t in dprimes if -10 < t[2] < 10)}")


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    loaded_specs = []
    loaded_specs.extend(CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-08-16 educated guesses.yaml")))
    loaded_specs.extend(CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-07-15 40k different decay.yaml")))
    loaded_specs.extend(CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications", "2021-06-25 search for more sensible parameters.yaml")))
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
        main(spec=spec,
             exclude_repeated_items=True,
             overwrite=False)

    logger.info("Done!")
