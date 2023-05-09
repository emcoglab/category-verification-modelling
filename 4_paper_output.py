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

import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from matplotlib import pyplot
from numpy import trapz, isnan, nan, mean, array
from numpy.random import seed
from pandas import DataFrame, Series, isna
from seaborn import jointplot, set_theme
from statsmodels.stats.inter_rater import fleiss_kappa

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import \
    CategoryVerificationParticipantOriginal, CategoryVerificationParticipantReplication, \
    CategoryVerificationItemDataOriginal, CategoryVerificationItemDataBlockedValidation, \
    CategoryVerificationParticipantBlockedValidation, CategoryVerificationItemDataReplication, \
    CategoryVerificationParticipantBalancedValidation, CategoryVerificationItemDataValidationBalanced
from framework.data.entities import CategoryObjectPair
from framework.data.col_names import ColNames
from framework.data.filter import Filter
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.evaluation.datasets import ParticipantDataset
from framework.evaluation.figures import opacity_for_overlap, named_colour, RGBA
from framework.evaluation.load import load_model_output_from_dir


# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")
OUTPUT_DIR = Path("/Users/caiwingfield/Resilio Sync/Lancaster/ Current/CV output")

# Shared
_n_threshold_steps = 10
THRESHOLDS = [i / _n_threshold_steps for i in range(_n_threshold_steps + 1)]  # linspace was causing weird float rounding errors

# Additional col names
MODEL_GUESS = "Model guessd"
MODEL_PEAK_ACTIVATION = "Model peak post-SOA activation"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class ParticipantPlotData:
    """Data to be used to plot participant performance ROC curves."""
    hit_rates: Series
    fa_rates: Series
    dataset_name: str
    colour: str
    symbol: str


def main(spec: CategoryVerificationJobSpec, exclude_repeated_items: bool,
         restrict_to_answerable_items: bool, validation_run: bool,
         participant_datasets: Optional[ParticipantDataset], items_matching_participant_dataset: ParticipantDataset,
         no_propagation: bool, overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    logger.info("")

    if participant_datasets is not None:
        assert participant_datasets == items_matching_participant_dataset

    # Determine directory paths with optional tests for early exit
    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    save_dir = Path(OUTPUT_DIR, spec.output_location_relative())
    if no_propagation:
        model_output_dir = Path(model_output_dir.parent, model_output_dir.name + "_no_propagation")
        save_dir = Path(save_dir.parent, save_dir.name + "_no_propagation")
    if validation_run:
        model_output_dir = Path(model_output_dir, "validation")
        save_dir = Path(save_dir, "validation")
    else:
        save_dir = Path(save_dir, "original")
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

    activation_traces_dir = Path(model_output_dir, "activation traces")

    if save_dir.exists() and not overwrite:
        logger.info(f"Evaluation complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    # Just for archiving purposes
    model_output_copy_dir = Path(save_dir, "activation traces")
    shutil.copytree(activation_traces_dir, model_output_copy_dir, dirs_exist_ok=True)  # Overwrite!

    if validation_run:
        trial_type_filter = None
    else:
        trial_type_filter = Filter.new_trial_type_filter({('test', True), ('filler', False)})
    if exclude_repeated_items:
        repeated_items_filter = Filter.new_repeated_item_filter(tokeniser=modified_word_tokenize)
    else:
        repeated_items_filter = None

    filter_sets: Dict[str, List[Filter | None]] = {
        "both": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["superordinate", "basic"]),
            trial_type_filter,
            repeated_items_filter,
        ],
        "superordinate": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["superordinate"]),
            trial_type_filter,
            repeated_items_filter,
        ],
        "basic": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["basic"]),
            trial_type_filter,
            repeated_items_filter,
        ],
    }

    # When validating, we can also break down by category domains
    if validation_run:
        new_filter_sets: Dict[str, List[Filter]] = filter_sets.copy()
        for name, filter_set in filter_sets.items():
            for category_domain in ["natural", "artefact"]:
                if category_domain == "natural" and name == "superordinate":
                    # There are no items here
                    continue
                new_filter_sets[f"{category_domain} {name}"] = filter_set + [Filter.new_category_domain_filter([category_domain])]
        filter_sets = new_filter_sets

    # Add model peak activations
    model_data: Dict[CategoryObjectPair, DataFrame] = load_model_output_from_dir(activation_traces_dir, validation=validation_run, for_participant_dataset=items_matching_participant_dataset)

    activation_plots_dir = Path(save_dir, "activation plots")
    activation_plots_dir.mkdir(parents=False, exist_ok=True)
    plot_activation_traces(model_data, spec=spec, save_to_dir=activation_plots_dir)

    items_name_fragment = get_item_name_fragment(items_matching_participant_dataset)
    participants_name_fragment = get_participant_name_fragment(participant_datasets, validation_run)

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'
    filename_prefix += f" {items_name_fragment}"
    if participants_name_fragment:
        filename_prefix += f" {participants_name_fragment}"

    items_df = get_items_df(items_matching_participant_dataset, validation_run)

    for filter_set_name, filter_set in filter_sets.items():

        filter_set: List[Filter] = list(filter_set)  # we may edit it

        if restrict_to_answerable_items:
            items_df[MODEL_PEAK_ACTIVATION] = items_df.apply(get_peak_activation, axis=1,
                                                             allow_missing_objects=False, model_data=model_data, spec=spec)
            filter_set.append(Filter(exclusion_selector=lambda row: isna(row[MODEL_PEAK_ACTIVATION]),
                                     name="available to model"))
        else:
            items_df[MODEL_PEAK_ACTIVATION] = items_df.apply(get_peak_activation, axis=1,
                                                             allow_missing_objects=True, model_data=model_data, spec=spec)

        save_item_exclusions(items_df, filter_set, Path(save_dir, f"{items_name_fragment} item exclusion for {filter_set_name}.csv"))

        filtered_df = Filter.apply_filters(filter_set, to_df=items_df)

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
        filename_suffix = filter_set_name

        # Participant hitrates
        participant_plot_datasets = []
        if validation_run:
            if participant_datasets in {ParticipantDataset.validation, ParticipantDataset.validation_plus_balanced}:
                participant_data = CategoryVerificationParticipantBlockedValidation()
                # TODO: don't just check it appears to work, verify this line is doing the right thing
                participant_summary_df = participant_data.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="validation", colour="forestgreen", symbol="+")
                )
            if participant_datasets in {ParticipantDataset.balanced, ParticipantDataset.validation_plus_balanced}:
                participant_data = CategoryVerificationParticipantBalancedValidation()
                participant_summary_df = participant_data.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="balanced", colour="lightseagreen", symbol="x")
                )

        else:
            if participant_datasets in {ParticipantDataset.original, ParticipantDataset.original_plus_replication}:
                participant_data = CategoryVerificationParticipantOriginal()
                participant_summary_df = participant_data.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="original", colour="blueviolet", symbol="+")
                )
            if participant_datasets in {ParticipantDataset.replication, ParticipantDataset.original_plus_replication}:
                participant_data = CategoryVerificationParticipantReplication()
                participant_summary_df = participant_data.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="replication", colour="mediumvioletred", symbol="x")
                )

        plot_roc(model_hit_rates, model_false_alarm_rates,
                 participant_plot_datasets,
                 filename_prefix, filename_suffix, save_dir,
                 model_colour="mediumblue",
                 participant_area_colour="indigo",
                 )

        plot_peak_activation_vs_affirmative_proportion(
            filtered_df,
            filename_prefix, filename_suffix, save_dir,
        )

        with Path(save_dir, f"{filename_prefix} data {filename_suffix}.csv") as f:
            filtered_df.to_csv(f, index=False)

    if participant_datasets is not None:
        agreement_path: Path = Path(save_dir, f"{filename_prefix} agreement.csv")
        participant_agreement(validation_run, participant_datasets, agreement_path, filter_responses_faster_than_ms=200)


def plot_activation_traces(model_data: Dict[CategoryObjectPair, DataFrame], spec: CategoryVerificationJobSpec, save_to_dir: Path) -> None:
    for cop, activation_data in model_data.items():
        object_label_linguistic, object_label_sensorimotor = substitutions_for(cop.object_label)
        object_label_linguistic_multiword_parts: List[str] = modified_word_tokenize(object_label_linguistic)

        set_theme(style="ticks", rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
        })

        fig, ax = pyplot.subplots()

        activation_data[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].plot.line(color="orange")
        for part in object_label_linguistic_multiword_parts:
            activation_data[OBJECT_ACTIVATION_LINGUISTIC_f.format(part)].plot.line(color="blue")

        ax.set_xlim([0, spec.run_for_ticks])
        ax.set_ylim([0, 1])

        ax.set_xlabel("Time")
        ax.set_ylabel("Activation")

        pyplot.savefig(Path(save_to_dir, f"{cop.category_label} -> {cop.object_label}.svg"), dpi=1200, bbox_inches="tight")
        pyplot.close(fig)


def get_peak_activation(row, *, allow_missing_objects: bool, spec: CategoryVerificationJobSpec, model_data: Dict[CategoryObjectPair, DataFrame]) -> float:
    cop = CategoryObjectPair(category_label=row[ColNames.CategoryLabel], object_label=row[ColNames.ImageObject])
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


def get_items_df(items_matching_participant_dataset, validation_run):
    # apply filters
    if validation_run:
        if items_matching_participant_dataset == ParticipantDataset.validation:
            items_df = CategoryVerificationItemDataBlockedValidation().data
        elif items_matching_participant_dataset == ParticipantDataset.balanced:
            items_df = CategoryVerificationItemDataValidationBalanced().data
        else:
            raise NotImplementedError()
    else:
        if items_matching_participant_dataset == ParticipantDataset.original:
            items_df = CategoryVerificationItemDataOriginal().data
        elif items_matching_participant_dataset == ParticipantDataset.replication:
            items_df = CategoryVerificationItemDataReplication().data
        elif items_matching_participant_dataset == ParticipantDataset.original_plus_replication:
            items_df = CategoryVerificationItemDataOriginal().data
            logger.warning(
                "Participant-related values not yet correct when using all participants, these will be omitted.")
            items_df.drop(columns=[ColNames.ResponseAccuracyMean,
                                   ColNames.ResponseAccuracySD,
                                   ColNames.ParticipantCount,
                                   ColNames.ResponseRTMean_ms,
                                   ColNames.ResponseRTSD_ms,
                                   ],
                          inplace=True)
        else:
            raise NotImplementedError()
    return items_df


def get_item_name_fragment(items_matching_participant_dataset):
    if items_matching_participant_dataset in {ParticipantDataset.original, ParticipantDataset.original_plus_replication,
                                              ParticipantDataset.replication}:
        items_name_fragment = "original items"
    elif items_matching_participant_dataset == ParticipantDataset.validation:
        items_name_fragment = "validation items"
    elif items_matching_participant_dataset == ParticipantDataset.balanced:
        items_name_fragment = "balanced items"
    else:
        raise NotImplementedError()
    return items_name_fragment


def get_participant_name_fragment(participant_datasets, validation_run):
    if validation_run:
        if participant_datasets == ParticipantDataset.validation_plus_balanced:
            participants_name_fragment = "all participants"
        elif participant_datasets == ParticipantDataset.validation:
            participants_name_fragment = "validation participants"
        elif participant_datasets == ParticipantDataset.balanced:
            participants_name_fragment = "balanced participants"
        elif participant_datasets is None:
            participants_name_fragment = ""
        else:
            raise ValueError(participant_datasets)
    else:
        if participant_datasets == ParticipantDataset.original_plus_replication:
            participants_name_fragment = "all participants"
        elif participant_datasets == ParticipantDataset.original:
            participants_name_fragment = "original participants"
        elif participant_datasets == ParticipantDataset.replication:
            participants_name_fragment = "replication participants"
        elif participant_datasets is None:
            participants_name_fragment = ""
        else:
            raise ValueError(participant_datasets)
    return participants_name_fragment


def save_item_exclusions(items_df: DataFrame, filters: List[Filter], save_path: Path):
    temp_df = items_df.copy()
    filters: List[Filter] = [f for f in filters if f is not None]
    for f in filters:
        temp_df = f.add_to_df(temp_df)
    with save_path.open("w") as filtered_items_file:
        temp_df.to_csv(filtered_items_file, index=False)


def participant_agreement(validation_run: bool, participant_datasets: ParticipantDataset, agreement_path: Path, filter_responses_faster_than_ms: Optional[float] = None):

    # Haven't done these yet!
    if not validation_run or  participant_datasets != ParticipantDataset.balanced:
        logger.warning("Skipping participant-agreement calculation for this dataset, not yet implemented!")
        return

    item_data = CategoryVerificationItemDataValidationBalanced()
    participant_data = CategoryVerificationParticipantBalancedValidation().data

    if filter_responses_faster_than_ms is not None:
        participant_data = participant_data[participant_data[ColNames.RT_ms] >= filter_responses_faster_than_ms]

    agreements = []
    for list_i, list_data in item_data.item_data_by_list.items():
        trials_in_list = participant_data.merge(
            right=list_data[[ColNames.CategoryLabel, ColNames.ImageObject, ColNames.List]],
            on=[ColNames.CategoryLabel, ColNames.ImageObject, ColNames.List],
            how="right")
        agreements.append({
            "List": list_i,
            "Fleiss' kappa": __compute_kappa(trials_in_list)
        })
    with agreement_path.open("w") as f:
        DataFrame(agreements).to_csv(f, index=False)


# todo: extract duplicate
def __compute_kappa(trials_in_list: DataFrame) -> float:
    trials_in_list = trials_in_list.copy()
    trials_in_list["yes"] = trials_in_list[ColNames.Response]
    trials_in_list["no"] = ~trials_in_list[ColNames.Response]
    # Needs to to have "subjects" (i.e. items) as rows and "categories" (i.e. responses) as columns, with cells
    # containing counts of ratings
    d = trials_in_list.groupby([ColNames.CategoryLabel, ColNames.ImageObject])[["yes", "no"]].sum()
    # Drop rows containing incomplete data
    d["either"] = d["yes"] + d["no"]
    d = d[d["either"] == d["either"].max()]
    # noinspection NonAsciiCharacters
    κ = fleiss_kappa(d[["yes", "no"]].to_numpy(), method="fleiss")
    return κ


def plot_peak_activation_vs_affirmative_proportion(df: DataFrame, filename_prefix: str, filename_suffix: str, save_dir: Path) -> None:
    set_theme(style="ticks", rc={
        "axes.spines.right": False,
        "axes.spines.top":   False,
    })

    g = jointplot(data=df, x=ColNames.ResponseAffirmativeProportion, y=MODEL_PEAK_ACTIVATION,
                  kind="reg", truncate=False,
                  marginal_kws={"kde": False})

    g.fig.savefig(Path(save_dir, f"{filename_prefix} model peak vs affirmative prop {filename_suffix}.png"), dpi=1200, bbox_inches='tight')
    g.fig.savefig(Path(save_dir, f"{filename_prefix} model peak vs affirmative prop {filename_suffix}.svg"), dpi=1200, bbox_inches='tight')
    pyplot.close(g.fig)


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

    set_theme(style="whitegrid")

    fig, ax = pyplot.subplots()

    # AUC
    auc = trapz(list(reversed(model_hit_rates)), list(reversed(model_fa_rates)))

    # Identity line
    identity_plot = pyplot.plot([0, 1], [0, 1], "r--",
                                label="Random classifier")
    # Model
    model_plot = pyplot.plot(model_fa_rates, model_hit_rates, "-", color=model_colour,
                             label="Model")

    legend_items = [identity_plot, model_plot]
    if participant_plot_datasets:
        participant_aucs = []
        individual_area_colour: RGBA = named_colour(
            participant_area_colour,
            with_alpha=opacity_for_overlap(desired_total_opacity=0.4,
                                           n_overlaps=sum(len(d.fa_rates) for d in participant_plot_datasets)))
        for participant_plot_data in participant_plot_datasets:
            # Participant points
            pyplot.plot(participant_plot_data.fa_rates, participant_plot_data.hit_rates,
                        participant_plot_data.symbol, color=participant_plot_data.colour,
                        label=f"Participants ({participant_plot_data.dataset_name} dataset)")
            # Participant mean spline interpolation
            # pyplot.plot(participant_interpolated_x, participant_interpolated_y, "g--")
            # Participant linearly interpolated areas
            for participant_fa, participant_hit in zip(participant_plot_data.fa_rates, participant_plot_data.hit_rates):
                px = [0, participant_fa, 1]
                py = [0, participant_hit, 1]
                pyplot.fill_between(px, py, color=individual_area_colour, label='_nolegend_')
                participant_aucs.append(trapz(py, px))

        ppt_title_clause = f"; " \
                           f"ppt min/mean/max:" \
                           f" [{min(participant_aucs):.3f}," \
                           f" {mean(array(participant_aucs)):.3f}," \
                           f" {max(participant_aucs):.3f}]"
    else:
        ppt_title_clause = ""

    # Style graph
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"ROC curve"
                 f" {filename_suffix}\n"
                 f"(AUC model:"
                 f" {auc:.3f}"
                 f"{ppt_title_clause})"
                 )
    ax.set_aspect('equal')
    pyplot.legend()

    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.png"), dpi=1200, bbox_inches='tight')
    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.svg"), dpi=1200, bbox_inches='tight')
    pyplot.close(fig)


@dataclass
class ArgSet:
    validation_run: bool
    participant_datasets: Optional[ParticipantDataset]
    items_matching_participant_dataset: ParticipantDataset

    exclude_repeated_items: bool = True
    restrict_to_answerable_items: bool = True

    overwrite: bool = True


# noinspection DuplicatedCode
if __name__ == '__main__':
    seed(1)  # Reproducible results

    logger.info("Running %s" % " ".join(sys.argv))

    arg_sets: List[ArgSet] = [
        ArgSet(validation_run=False, participant_datasets=None, items_matching_participant_dataset=ParticipantDataset.original_plus_replication),
        ArgSet(validation_run=False, participant_datasets=ParticipantDataset.original, items_matching_participant_dataset=ParticipantDataset.original),
        ArgSet(validation_run=False, participant_datasets=ParticipantDataset.replication, items_matching_participant_dataset=ParticipantDataset.replication),
        ArgSet(validation_run=False, participant_datasets=ParticipantDataset.original_plus_replication, items_matching_participant_dataset=ParticipantDataset.original_plus_replication),
        ArgSet(validation_run=True,  participant_datasets=None, items_matching_participant_dataset=ParticipantDataset.validation),
        ArgSet(validation_run=True,  participant_datasets=None, items_matching_participant_dataset=ParticipantDataset.balanced),
        ArgSet(validation_run=True,  participant_datasets=ParticipantDataset.validation, items_matching_participant_dataset=ParticipantDataset.validation),
        ArgSet(validation_run=True,  participant_datasets=ParticipantDataset.balanced, items_matching_participant_dataset=ParticipantDataset.balanced),
    ]

    loaded_specs = CategoryVerificationJobSpec.load_multiple(Path(Path(__file__).parent,
                                                                  "job_specifications",
                                                                  "2023-01-12 Paper output.yaml"))
    cca_spec: CategoryVerificationJobSpec = loaded_specs[0]
    no_cca_spec: CategoryVerificationJobSpec = loaded_specs[1]

    for arg_set in arg_sets:
        main(spec=cca_spec, no_propagation=False, **asdict(arg_set))
        main(spec=cca_spec, no_propagation=True, **asdict(arg_set))
        main(spec=no_cca_spec, no_propagation=False, **asdict(arg_set))
        main(spec=no_cca_spec, no_propagation=True, **asdict(arg_set))

    logger.info("Done!")
