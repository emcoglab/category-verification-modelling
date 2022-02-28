#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Analysing the output of one model in detail.
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
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Dict

from matplotlib import pyplot
from numpy.random import seed
from pandas import DataFrame

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import CategoryVerificationParticipantOriginal, \
    CategoryVerificationItemData, CategoryObjectPair, ColNames
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.evaluation.decision import make_model_decision, Outcome, make_all_model_decisions, DecisionColNames
from framework.evaluation.load import load_model_output_from_dir

_logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")

# Shared
CV_SUBJECT_DATA: CategoryVerificationParticipantOriginal = CategoryVerificationParticipantOriginal()
THRESHOLDS = [i / 100 for i in range(101)]  # linspace was causing weird float rounding errors


def plot_object_activation_traces(spec: CategoryVerificationJobSpec,
                                  all_model_data: Dict[CategoryObjectPair, DataFrame],
                                  decision_threshold_yes: float, decision_threshold_no: float,
                                  save_dir: Path):
    """
    Plot activation traces for object activations in HIT, MISS, FA and CR categories
    """

    fig, ((ax_hit, ax_fa), (ax_miss, ax_cr)) = pyplot.subplots(2, 2,
                                                               sharex='all', sharey='all',
                                                               figsize=(21, 14))
    ax_hit.set_title("HIT")
    ax_miss.set_title("MISS")
    ax_fa.set_title("FA")
    ax_cr.set_title("CR")
    alpha = 50
    linewidth = 2
    sm_colour = f"#ff8800{alpha}"
    ling_colour = f"#0088ff{alpha}"
    ref_colour = f"#000000{alpha}"
    for (category_label, object_label), activation_df in all_model_data.items():

        object_label_linguistic, object_label_sensorimotor = substitutions_for(object_label)

        model_decision, decision_made_at_time, component = make_model_decision(
            object_label,
            decision_threshold_no, decision_threshold_yes,
            activation_df,
            spec)

        model_outcome: Outcome = Outcome.from_decision(
            decision=model_decision,
            should_be_yes=CategoryVerificationItemData().is_correct(category_label, object_label))

        # Determine correct axis
        if model_outcome == Outcome.Hit:
            ax = ax_hit
        elif model_outcome == Outcome.Miss:
            ax = ax_miss
        elif model_outcome == Outcome.FalseAlarm:
            ax = ax_fa
        elif model_outcome == Outcome.CorrectRejection:
            ax = ax_cr
        else:
            raise ValueError()
        # Plot lines on this axis
        df = activation_df[activation_df.index <= decision_made_at_time]
        ax.plot(df.index.values,
                df[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].values,
                color=sm_colour, linewidth=linewidth)
        if component is not None and component == component.sensorimotor:
            # Plot decision dot
            ax.scatter(df.index.values[-1],
                       df[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].values[-1],
                       marker='o', color=sm_colour)
        for a in modified_word_tokenize(object_label_linguistic):
            ax.plot(df.index.values,
                    df[OBJECT_ACTIVATION_LINGUISTIC_f.format(a)].values,
                    color=ling_colour, linewidth=linewidth)
            if component is not None and component == component.linguistic:
                # Plot decision dot
                # TODO: when there are multiple items and only one of them contributed to the decision, we're still
                #  plotting dots on all of them, which means the dots can end up in weird places
                ax.scatter(df.index.values[-1], df[OBJECT_ACTIVATION_LINGUISTIC_f.format(a)].values[-1],
                           marker='o', color=ling_colour)
    # Plot reference lines
    for ax in [ax_hit, ax_miss, ax_fa, ax_cr]:
        ax.axhline(y=decision_threshold_yes, linewidth=linewidth, color=ref_colour)
        ax.axhline(y=decision_threshold_no, linewidth=linewidth, color=ref_colour)
    pyplot.tight_layout()
    fig_path = Path(save_dir, f"activations {spec.shorthand}.png")
    fig.savefig(fig_path.as_posix())
    _logger.info(f"Saved activations to {fig_path.as_posix()}")
    pyplot.close(fig)


def categorise_errors(spec: CategoryVerificationJobSpec,
                      all_model_data: Dict[CategoryObjectPair, DataFrame],
                      decision_threshold_yes: float, decision_threshold_no: float):
    """
    Categories the model's correct and incorrect guesses by the type of the stimulus
    """

    model_guesses_df = make_all_model_decisions(all_model_data, decision_threshold_yes, decision_threshold_no, spec)

    # Add taxonomic level for all items
    model_guesses_df = model_guesses_df.merge(
        CategoryVerificationItemData().dataframe[[
            ColNames.CategoryLabel, ColNames.ImageObject,
            # New columns to include
            ColNames.TaxonomicLevel, ColNames.EasyHardToReject,
        ]],
        on=[ColNames.CategoryLabel, ColNames.ImageObject],
        how="left")

    easy_guesses = model_guesses_df[model_guesses_df[ColNames.EasyHardToReject] == "easy"]
    hard_guesses = model_guesses_df[model_guesses_df[ColNames.EasyHardToReject] == "hard"]

    superordinate_guesses = model_guesses_df[model_guesses_df[ColNames.TaxonomicLevel] == "superordinate"]
    basic_guesses         = model_guesses_df[model_guesses_df[ColNames.TaxonomicLevel] == "basic"]
    subordinate_guesses   = model_guesses_df[model_guesses_df[ColNames.TaxonomicLevel] == "subordinate"]

    probability_error_easy = len(easy_guesses[easy_guesses[DecisionColNames.ModelIsCorrect] == False]) / len(easy_guesses)
    probability_error_hard = len(hard_guesses[hard_guesses[DecisionColNames.ModelIsCorrect] == False]) / len(hard_guesses)

    probability_error_superordinate = len(superordinate_guesses[superordinate_guesses[DecisionColNames.ModelIsCorrect] == False]) / len(superordinate_guesses)
    probability_error_basic         = len(basic_guesses[basic_guesses[DecisionColNames.ModelIsCorrect] == False]) / len(basic_guesses)
    probability_error_subordinate   = len(subordinate_guesses[subordinate_guesses[DecisionColNames.ModelIsCorrect] == False]) / len(subordinate_guesses)

    _logger.info(f"Probability of model error on easy items: {probability_error_easy}")
    _logger.info(f"Probability of model error on hard items: {probability_error_hard}")

    _logger.info(f"Probability of model error on superordinate items: {probability_error_superordinate}")
    _logger.info(f"Probability of model error on basic items: {probability_error_basic}")
    _logger.info(f"Probability of model error on subordinate items: {probability_error_subordinate}")


def main(spec: CategoryVerificationJobSpec, decision_threshold_yes: float, decision_threshold_no: float,
         exclude_repeated_items: bool, overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    assert decision_threshold_no < decision_threshold_yes

    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    if not model_output_dir.exists():
        _logger.warning(f"Model output not found for v{VERSION} in directory {model_output_dir.as_posix()}")
        return
    if not Path(model_output_dir, " MODEL RUN COMPLETE").exists():
        _logger.info(f"Incomplete model run found in {model_output_dir.as_posix()}")
        return

    # Output dir
    save_dir = Path(model_output_dir, " output")  # TODO: this is a bad directory name; unexpected, ambiguous
    if save_dir.exists() and not overwrite:
        _logger.info(f"Output complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=False, exist_ok=True)

    # Load output data data from this run
    try:
        all_model_data = load_model_output_from_dir(model_output_dir, exclude_repeated_items=exclude_repeated_items)
    except FileNotFoundError:
        _logger.warning(f"No model data in {model_output_dir.as_posix()}")
        return

    plot_object_activation_traces(spec, all_model_data, decision_threshold_yes, decision_threshold_no, save_dir)

    categorise_errors(spec, all_model_data, decision_threshold_yes, decision_threshold_no)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    _logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    spec = CategoryVerificationJobSpec.load_multiple(
        Path(Path(__file__).parent, "job_specifications",
             "2022-01-24 More variations on the current favourite.yaml"))[0]  # Just the first one from this batch
    yes, no = 0.4, 0.2
    main(spec=spec,
         decision_threshold_yes=yes, decision_threshold_no=no,
         exclude_repeated_items=True,
         overwrite=True)

    _logger.info("Done!")
