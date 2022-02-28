"""
===========================
Common code for evaluation.
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

from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from numpy import nan
from pandas import DataFrame

from framework.maths import z_score
from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue, Component
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.data.category_verification_data import ColNames, CategoryVerificationItemData, CategoryObjectPair
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f


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
        # When it's undecided or waiting, we default to no.
        if (decision == Decision.Undecided) or (decision == Decision.Waiting):
            decision = Decision.No
        # This currently has no effect on what is returned by this function as we only care about yesses

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


class DecisionColNames:
    """Additional ColNames for model decision data"""
    ModelDecision: str  = "Model decision"
    DecisionTime: str   = "Decision made at time"
    ModelOutcome: str   = "Model outcome"
    ModelIsCorrect: str = "Model is correct",


def make_model_decision(object_label, decision_threshold_no, decision_threshold_yes, model_data, spec) -> Tuple[Decision, int, Optional[Component]]:
    """Make a decision for this object label."""

    object_label_linguistic, object_label_sensorimotor = substitutions_for(object_label)
    object_label_linguistic_multiword_parts: List[str] = modified_word_tokenize(object_label_linguistic)

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

        # Return decision when made: either component can make either decision
        if sensorimotor_decision.made:
            return sensorimotor_decision, tick, Component.sensorimotor
        for decision in linguistic_decisions:
            if decision.made:
                return decision, tick, Component.linguistic
    # If we run out of time
    return Decision.Undecided, spec.run_for_ticks, None


def make_all_model_decisions(all_model_data, decision_threshold_yes, decision_threshold_no, spec) -> DataFrame:
    """Make decisions for all stimuli."""

    model_guesses = []
    for category_label, object_label in CategoryVerificationItemData().category_object_pairs():
        item_is_of_category: bool = CategoryVerificationItemData().is_correct(category_label, object_label)

        try:
            model_data = all_model_data[CategoryObjectPair(category_label, object_label)]
        # No model output was saved
        except KeyError:
            continue

        model_decision: Decision
        decision_made_at_time: int
        model_decision, decision_made_at_time, _component = make_model_decision(
            object_label,
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
        DecisionColNames.ModelDecision, DecisionColNames.DecisionTime,
        DecisionColNames.ModelOutcome, DecisionColNames.ModelIsCorrect,
    ])
    return model_guesses_df


def performance_for_thresholds(all_model_data: Dict[CategoryObjectPair, DataFrame],
                               restrict_to_answerable_items: bool,
                               decision_threshold_yes: ActivationValue, decision_threshold_no: ActivationValue,
                               loglinear: bool,
                               spec: CategoryVerificationJobSpec, save_dir: Path) -> Tuple[float, float, float]:
    """
    Returns correct_rate and dprime and criterion.

    :param loglinear: use the loglinear transform for computing d' and criterion (but not for hitrate).
    """

    ground_truth_dataframe = CategoryVerificationItemData().dataframe

    model_guesses_df = make_all_model_decisions(all_model_data, decision_threshold_yes, decision_threshold_no, spec)

    # Format columns
    model_guesses_df[DecisionColNames.DecisionTime] = model_guesses_df[DecisionColNames.DecisionTime].astype('Int64')

    results_dataframe = ground_truth_dataframe.merge(model_guesses_df[[
        ColNames.CategoryLabel, ColNames.ImageObject,
        # Exclude ColNames.ShouldBeVerified so we don't get duplicated columns on the merge
        DecisionColNames.ModelDecision, DecisionColNames.DecisionTime,
        DecisionColNames.ModelOutcome, DecisionColNames.ModelIsCorrect,
    ]], how="left", on=[ColNames.CategoryLabel, ColNames.ImageObject])

    # Save individual threshold data for verification
    save_dir.mkdir(parents=False, exist_ok=True)
    with Path(save_dir, f"no{decision_threshold_no}_yes{decision_threshold_yes}.csv").open("w") as f:
        results_dataframe.to_csv(f, header=True, index=False)

    model_hit_count = len(model_guesses_df[model_guesses_df[DecisionColNames.ModelOutcome] == Outcome.Hit.name])
    model_fa_count = len(model_guesses_df[model_guesses_df[DecisionColNames.ModelOutcome] == Outcome.FalseAlarm.name])
    model_correct_count = len(model_guesses_df[model_guesses_df[DecisionColNames.ModelIsCorrect] == True])

    if restrict_to_answerable_items:
        # Rates computed as a proportion of trials on which the model can decide
        n_trials_signal = len(model_guesses_df[model_guesses_df[ColNames.ShouldBeVerified] == True])
        n_trials_noise = len(model_guesses_df[model_guesses_df[ColNames.ShouldBeVerified] == False])
    else:
        # Rates computed as proportion of ALL trials, not all trials on which the model can decide
        n_trials_signal = len(results_dataframe[results_dataframe[ColNames.ShouldBeVerified] == True])
        n_trials_noise = len(results_dataframe[results_dataframe[ColNames.ShouldBeVerified] == False])

    model_correct_rate = model_correct_count / (n_trials_signal + n_trials_noise)

    if loglinear:
        # See Stanislav & Todorov (1999, BRMIC)
        model_hit_count += 0.5
        model_fa_count += 0.5
        n_trials_signal += 1
        n_trials_noise += 1

    model_hit_rate = model_hit_count / n_trials_signal
    model_false_alarm_rate = model_fa_count / n_trials_noise

    # This is a simple Y/N task, not a 2AFC, so we can just use standard d-prime
    if (model_hit_rate == 0) or (model_false_alarm_rate == 1) or (model_false_alarm_rate == 0) or (model_hit_rate == 1):
        # Can't compute a dprime or a criterion, so mark as missing
        model_dprime = nan
        model_criterion = nan
    else:
        model_dprime = z_score(model_hit_rate) - z_score(model_false_alarm_rate)
        model_criterion = - (z_score(model_hit_rate) + z_score(model_false_alarm_rate)) / 2

    return model_correct_rate, model_dprime, model_criterion
