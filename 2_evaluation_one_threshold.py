#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Evaluating the combined model script for category verification task using the
one-threshold decision procedure.
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

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from matplotlib import pyplot
from numpy import trapz, isnan, nan
from numpy.random import seed
from pandas import DataFrame, Series

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import ColNames, CategoryObjectPair, Filter, \
    CategoryVerificationParticipantOriginal, CategoryVerificationParticipantReplication, \
    CategoryVerificationItemData, CategoryVerificationItemDataBlockedValidation, \
    CategoryVerificationParticipantBlockedValidation, CategoryVerificationItemDataReplication
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.evaluation.figures import opacity_for_overlap, named_colour, RGBA
from framework.evaluation.load import load_model_output_from_dir


# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")

# Shared
_n_threshold_steps = 10
THRESHOLDS = [i / _n_threshold_steps for i in range(_n_threshold_steps + 1)]  # linspace was causing weird float rounding errors

# Additional col names
MODEL_GUESS = "Model guessd"
MODEL_PEAK_ACTIVATION = "Model peak post-SOA activation"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# noinspection PyArgumentList
# this is an IDE bug!
class ParticipantDatasetSelection(Enum):
    """Which participant dataset to use with ROC plotting."""
    # Initial experiment
    original = auto()  # Original participant set
    replication = auto()  # Replication participant set
    all = auto()  # Original and replication participant set
    # Validation experiment
    validation = auto()  # Validation participant set


@dataclass
class ParticipantPlotData:
    """Data to be used to plot participant performance ROC curves."""
    hit_rates: Series
    fa_rates: Series
    dataset_name: str
    colour: str


def main(spec: CategoryVerificationJobSpec, spec_filename: str, exclude_repeated_items: bool,
         restrict_to_answerable_items: bool, use_assumed_object_label: bool, validation_run: bool,
         participant_datasets: Optional[ParticipantDatasetSelection],
         no_propagation: bool, overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    logger.info("")
    logger.info(f"Spec: {spec_filename}")

    # Determine directory paths with optional tests for early exit
    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    if no_propagation:
        model_output_dir = Path(model_output_dir.parent, model_output_dir.name + "_no_propagation")
    if validation_run:
        model_output_dir = Path(model_output_dir, "validation")
    if not model_output_dir.exists():
        logger.warning(f"Model output not found for v{VERSION} in directory {model_output_dir.as_posix()}")
        return
    complete_file = Path(model_output_dir, " MODEL RUN COMPLETE")
    if not complete_file.exists():
        # Repair parallelised run completion files
        if not all(Path(complete_file.parent, complete_file.name + letter).exists() for letter in ALPHABET.lower()):
            for letter1 in ALPHABET.lower():
                if all(Path(complete_file.parent, complete_file.name + letter1 + "_" + letter2).exists() for letter2 in ALPHABET.lower()):
                    Path(complete_file.parent, complete_file.name + letter1).touch()
        if all(Path(complete_file.parent, complete_file.name + letter).exists() for letter in ALPHABET.lower()):
            logger.info(f"Parallelised model was complete, creating {complete_file.as_posix()}")
            complete_file.touch()
    if not complete_file.exists():
        logger.warning(f"Skipping incomplete model run: {complete_file.parent.as_posix()}")
        return
    save_dir = Path(model_output_dir, " evaluation")
    if save_dir.exists() and not overwrite:
        logger.info(f"Evaluation complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=False, exist_ok=True)

    if validation_run:
        trial_types = None
    else:
        trial_types = [('test', True), ('filler', False)]

    filters: List[Filter] = [
        Filter(
            name="superordinate" if not use_assumed_object_label else "superordinate (assumed image label)",
            category_taxonomic_levels=["superordinate"],
            trial_types=trial_types,
            repeated_items_tokeniser=modified_word_tokenize if exclude_repeated_items else None,
            use_assumed_object_label=use_assumed_object_label and exclude_repeated_items),
        Filter(
            name="basic" if not use_assumed_object_label else "basic (assumed image label)",
            category_taxonomic_levels=["basic"],
            trial_types=trial_types,
            repeated_items_tokeniser=modified_word_tokenize if exclude_repeated_items else None,
            use_assumed_object_label=use_assumed_object_label and exclude_repeated_items),
        Filter(
            name="both" if not use_assumed_object_label else "both (assumed image label)",
            category_taxonomic_levels=["superordinate", "basic"],
            trial_types=trial_types,
            repeated_items_tokeniser=modified_word_tokenize if exclude_repeated_items else None,
            use_assumed_object_label=use_assumed_object_label and exclude_repeated_items),
    ]

    # When validating, we can break down by category domains
    if validation_run:
        category_domains = ["natural", "artefact"]
        original_filters = list(filters)
        for f in original_filters:
            for category_domain in category_domains:
                new_filter = deepcopy(f)
                new_filter.category_domain = [category_domain]
                new_filter.name = f"{category_domain} {f.name}"
                if new_filter.category_domain == ["natural"] and new_filter.category_taxonomic_levels == ["superordinate"]:
                    # There are no items here
                    # TODO: this convolution seems a little silly
                    continue
                filters.append(new_filter)

    # Add model peak activations
    model_data: Dict[CategoryObjectPair, DataFrame] = load_model_output_from_dir(model_output_dir, validation=validation_run, use_assumed_object_label=use_assumed_object_label)

    def get_peak_activation(row, *, allow_missing_objects: bool) -> float:
        item_col = ColNames.ImageLabelAssumed if use_assumed_object_label else ColNames.ImageObject
        cop = CategoryObjectPair(category_label=row[ColNames.CategoryLabel], object_label=row[item_col])
        try:
            model_activations_df: DataFrame = model_data[cop]
        except KeyError:
            return nan
        # The decision rests on the peak activation over object labels over both components, so we can just take the max
        # of all of them
        object_label_linguistic, object_label_sensorimotor = substitutions_for(cop.object_label)
        object_label_linguistic_multiword_parts: List[str] = modified_word_tokenize(object_label_linguistic)
        # We are only interested in the activation after ths SOA
        post_soa_df = model_activations_df.loc[spec.soa_ticks+1:spec.run_for_ticks]

        peak_activation_sensorimotor = post_soa_df[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].max()

        linguistic_activations = [
            post_soa_df[OBJECT_ACTIVATION_LINGUISTIC_f.format(part)].max()
            for part in object_label_linguistic_multiword_parts
        ]

        if not allow_missing_objects:
            if isnan(peak_activation_sensorimotor) or any(isnan(a) for a in linguistic_activations):
                return nan

        if all(isnan(a) for a in linguistic_activations):
            # Can't take a max
            peak_activation_linguistic = nan
        else:
            peak_activation_linguistic = max(a for a in linguistic_activations if not isnan(a))

        if isnan(peak_activation_linguistic) and isnan(peak_activation_sensorimotor):
            # Can't take a max
            return nan
        else:
            return max(ac for ac in [peak_activation_linguistic, peak_activation_sensorimotor] if not isnan(ac))

    for cv_filter in filters:

        # apply filters
        if validation_run:
            filtered_df = CategoryVerificationItemDataBlockedValidation().dataframe_filtered(cv_filter)
        elif participant_datasets == ParticipantDatasetSelection.original:
            filtered_df = CategoryVerificationItemData().dataframe_filtered(cv_filter)
        elif participant_datasets == ParticipantDatasetSelection.replication:
            filtered_df = CategoryVerificationItemDataReplication().dataframe_filtered(cv_filter)
        elif participant_datasets == ParticipantDatasetSelection.all:
            filtered_df = CategoryVerificationItemData().dataframe_filtered(cv_filter)
            logger.warn("Participant-related values not yet correct when using all participants, these will be omitted.")
            filtered_df.drop(columns=[ColNames.ResponseAccuracyMean,
                                      ColNames.ResponseAccuracySD,
                                      ColNames.ParticipantCount,
                                      ColNames.ResponseRTMean,
                                      ColNames.ResponseRTSD,
                                      ],
                             inplace=True)
        else:
            raise NotImplementedError()

        if restrict_to_answerable_items:
            filtered_df[MODEL_PEAK_ACTIVATION] = filtered_df.apply(get_peak_activation, axis=1, allow_missing_objects=False)
            filtered_df.dropna(subset=[MODEL_PEAK_ACTIVATION], inplace=True)
        else:
            filtered_df[MODEL_PEAK_ACTIVATION] = filtered_df.apply(get_peak_activation, axis=1, allow_missing_objects=True)

        # Model hitrates
        model_hit_rates = []
        model_false_alarm_rates = []
        for decision_threshold in THRESHOLDS:

            hit_rate, fa_rate = performance_for_one_threshold_simplified(
                all_data=filtered_df,
                decision_threshold=decision_threshold,
                strict_inequality=True)
            model_hit_rates.append(hit_rate)
            model_false_alarm_rates.append(fa_rate)

        filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'
        if validation_run:
            if participant_datasets == ParticipantDatasetSelection.validation:
                filename_prefix += " validation participants"
            elif participant_datasets is not None:
                raise ValueError(participant_datasets)
        else:
            if participant_datasets == ParticipantDatasetSelection.all:
                filename_prefix += " all participants"
            elif participant_datasets == ParticipantDatasetSelection.original:
                filename_prefix += " original participants"
            elif participant_datasets == ParticipantDatasetSelection.replication:
                filename_prefix += " replication participants"
            elif participant_datasets is not None:
                raise ValueError(participant_datasets)
        filename_suffix = cv_filter.name

        # Participant hitrates
        participant_plot_datasets = []
        if validation_run and participant_datasets == ParticipantDatasetSelection.validation:
                # TODO: don't just check it works, verify this line is doing the right thing
            participant_dataset = CategoryVerificationParticipantBlockedValidation()
            participant_summary_df = participant_dataset.participant_summary_dataframe(
                use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                    filtered_df))
            participant_plot_datasets.append(
                ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                    fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                    dataset_name="validation", colour="forestgreen")
            )

        else:
            if participant_datasets in {ParticipantDatasetSelection.all, ParticipantDatasetSelection.original}:
                participant_dataset = CategoryVerificationParticipantOriginal()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemData.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="original", colour="blueviolet")
                )
            if participant_datasets in {ParticipantDatasetSelection.all, ParticipantDatasetSelection.replication}:
                participant_dataset = CategoryVerificationParticipantReplication()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemData.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="replication", colour="mediumvioletred")
                )

        plot_roc(model_hit_rates, model_false_alarm_rates,
                 participant_plot_datasets,
                 filename_prefix, filename_suffix, save_dir,
                 model_colour="mediumblue",
                 participant_area_colour="indigo",
                 )

        with Path(save_dir, f"{filename_prefix} data {filename_suffix}.csv") as f:
            filtered_df.to_csv(f, index=False)


def performance_for_one_threshold_simplified(
        all_data: DataFrame,
        decision_threshold: ActivationValue,
        strict_inequality: bool) -> Tuple[float, float]:
    """
    Returns hit_rate, false-alarm rate.
    """

    all_data = all_data.copy()  # So we can modify this local copy

    # Using strict inequality.
    # If using non-strict inequality, and if threshold == FULL_ACTIVATION, need to reduce it by 1e-10 to account for
    # floating point arithmetic.
    if strict_inequality:
        all_data[MODEL_GUESS] = all_data[MODEL_PEAK_ACTIVATION] > decision_threshold
    else:
        if decision_threshold == FULL_ACTIVATION:
            decision_threshold -= 1e-10
        all_data[MODEL_GUESS] = all_data[MODEL_PEAK_ACTIVATION] >= decision_threshold

    n_trials_signal = len(all_data[all_data[ColNames.ShouldBeVerified] == True])
    n_trials_noise = len(all_data[all_data[ColNames.ShouldBeVerified] == False])
    model_hit_count = len(all_data[(all_data[ColNames.ShouldBeVerified] == True) & (all_data[MODEL_GUESS] == True)])
    model_fa_count = len(all_data[(all_data[ColNames.ShouldBeVerified] == False) & (all_data[MODEL_GUESS] == True)])

    model_hit_rate = model_hit_count / n_trials_signal
    model_false_alarm_rate = model_fa_count / n_trials_noise

    return model_hit_rate, model_false_alarm_rate


def plot_roc(model_hit_rates, model_fa_rates,
             participant_plot_datasets: List[ParticipantPlotData],
             filename_prefix, filename_suffix, save_dir,
             model_colour: str, participant_area_colour: str):

    fig, ax = pyplot.subplots()

    # AUC
    auc = trapz(list(reversed(model_hit_rates)), list(reversed(model_fa_rates)))

    # Identity line
    pyplot.plot([0, 1], [0, 1], "r--")
    # Model
    pyplot.plot(model_fa_rates, model_hit_rates, "-", color=model_colour)

    legend_items = ["Random classifier", "Model"]
    if participant_plot_datasets:
        participant_aucs = []
        individual_area_colour: RGBA = named_colour(
            participant_area_colour,
            with_alpha=opacity_for_overlap(desired_total_opacity=0.4,
                                           n_overlaps=sum(len(d.fa_rates) for d in participant_plot_datasets)))
        for participant_plot_data in participant_plot_datasets:
            # Participant points
            pyplot.plot(participant_plot_data.fa_rates, participant_plot_data.hit_rates,
                        "+", color=participant_plot_data.colour)
            # Participant mean spline interpolation
            # pyplot.plot(participant_interpolated_x, participant_interpolated_y, "g--")
            # Participant linearly interpolated areas
            for participant_fa, participant_hit in zip(participant_plot_data.fa_rates, participant_plot_data.hit_rates):
                px = [0, participant_fa, 1]
                py = [0, participant_hit, 1]
                pyplot.fill_between(px, py, color=individual_area_colour, label='_nolegend_')
                participant_aucs.append(trapz(py, px))

            legend_items.append(f"Participants ({participant_plot_data.dataset_name} dataset)")

        ppt_title_clause = f"; " \
                           f"ppt range:" \
                           f" [{min(participant_aucs):.2f}," \
                           f" {max(participant_aucs):.2f}]"
    else:
        ppt_title_clause = ""

    # Style graph
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"ROC curve"
                 f" {filename_suffix}\n"
                 f"(AUC model:"
                 f" {auc:.2}"
                 f"{ppt_title_clause})"
                 )
    pyplot.legend(legend_items)

    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.png", dpi=1200))
    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.svg", dpi=1200))
    pyplot.close(fig)


# noinspection DuplicatedCode
if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    loaded_specs = []
    for sfn in [
        "2022-08-20 good roc model with cut connections.yaml",
    ]:
        loaded_specs.extend([(s, sfn, i) for i, s in enumerate(CategoryVerificationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications", sfn)))])

    for j, (spec, sfn, i) in enumerate(loaded_specs, start=1):
        logger.info(f"Evaluating model {j} of {len(loaded_specs)}")
        for no_propagation in [True, False]:
            main(
                spec=spec,
                spec_filename=f"{sfn} [{i}]",
                exclude_repeated_items=True,
                restrict_to_answerable_items=True,
                use_assumed_object_label=False,
                validation_run=True,
                participant_datasets=ParticipantDatasetSelection.validation,
                overwrite=True,
                no_propagation=no_propagation,
            )

    logger.info("Done!")
