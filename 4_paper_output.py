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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from matplotlib import pyplot
from numpy import trapz, isnan, nan
from numpy.random import seed
from pandas import DataFrame, Series
from seaborn import jointplot, set_theme
from statsmodels.stats.inter_rater import fleiss_kappa

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import ColNames, CategoryObjectPair, Filter, \
    CategoryVerificationParticipantOriginal, CategoryVerificationParticipantReplication, \
    CategoryVerificationItemDataOriginal, CategoryVerificationItemDataBlockedValidation, \
    CategoryVerificationParticipantBlockedValidation, CategoryVerificationItemDataReplication, \
    CategoryVerificationParticipantBalancedValidation, CategoryVerificationItemDataValidationBalanced
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
         restrict_to_answerable_items: bool, use_assumed_object_label: bool, validation_run: bool,
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
    if save_dir.exists() and not overwrite:
        logger.info(f"Evaluation complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    if validation_run:
        trial_types = None
    else:
        trial_types = [('test', True), ('filler', False)]

    filters: List[Filter] = [
        Filter(
            name="both" if not use_assumed_object_label else "both (assumed image label)",
            category_taxonomic_levels=["superordinate", "basic"],
            trial_types=trial_types,
            repeated_items_tokeniser=modified_word_tokenize if exclude_repeated_items else None,
            use_assumed_object_label=use_assumed_object_label and exclude_repeated_items),
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
    model_data: Dict[CategoryObjectPair, DataFrame] = load_model_output_from_dir(model_output_dir, validation=validation_run, for_participant_dataset=items_matching_participant_dataset, use_assumed_object_label=use_assumed_object_label)

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

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'

    if items_matching_participant_dataset in {ParticipantDataset.original, ParticipantDataset.original_plus_replication, ParticipantDataset.replication}:
        filename_prefix += " original items"
    elif items_matching_participant_dataset == ParticipantDataset.validation:
        filename_prefix += " validation items"
    elif items_matching_participant_dataset == ParticipantDataset.balanced:
        filename_prefix += " balanced items"
    else:
        raise NotImplementedError()

    if validation_run:
        if participant_datasets == ParticipantDataset.validation_plus_balanced:
            filename_prefix += " all participants"
        elif participant_datasets == ParticipantDataset.validation:
            filename_prefix += " validation participants"
        elif participant_datasets == ParticipantDataset.balanced:
            filename_prefix += " balanced participants"
        elif participant_datasets is not None:
            raise ValueError(participant_datasets)
    else:
        if participant_datasets == ParticipantDataset.original_plus_replication:
            filename_prefix += " all participants"
        elif participant_datasets == ParticipantDataset.original:
            filename_prefix += " original participants"
        elif participant_datasets == ParticipantDataset.replication:
            filename_prefix += " replication participants"
        elif participant_datasets is not None:
            raise ValueError(participant_datasets)

    for cv_filter in filters:

        # apply filters
        if validation_run:
            if items_matching_participant_dataset == ParticipantDataset.validation:
                filtered_df = CategoryVerificationItemDataBlockedValidation().data_filtered(cv_filter)
            elif items_matching_participant_dataset == ParticipantDataset.balanced:
                filtered_df = CategoryVerificationItemDataValidationBalanced().data_filtered(cv_filter)
            else:
                raise NotImplementedError()
        else:
            if items_matching_participant_dataset == ParticipantDataset.original:
                filtered_df = CategoryVerificationItemDataOriginal().data_filtered(cv_filter)
            elif items_matching_participant_dataset == ParticipantDataset.replication:
                filtered_df = CategoryVerificationItemDataReplication().data_filtered(cv_filter)
            elif items_matching_participant_dataset == ParticipantDataset.original_plus_replication:
                filtered_df = CategoryVerificationItemDataOriginal().data_filtered(cv_filter)
                logger.warning("Participant-related values not yet correct when using all participants, these will be omitted.")
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
        filename_suffix = cv_filter.name

        # Participant hitrates
        participant_plot_datasets = []
        if validation_run:
            if participant_datasets in {ParticipantDataset.validation, ParticipantDataset.validation_plus_balanced}:
                participant_dataset = CategoryVerificationParticipantBlockedValidation()
                # TODO: don't just check it appears to work, verify this line is doing the right thing
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="validation", colour="forestgreen", symbol="+")
                )
            if participant_datasets in {ParticipantDataset.balanced, ParticipantDataset.validation_plus_balanced}:
                participant_dataset = CategoryVerificationParticipantBalancedValidation()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="balanced", colour="lightseagreen", symbol="x")
                )

        else:
            if participant_datasets in {ParticipantDataset.original, ParticipantDataset.original_plus_replication}:
                participant_dataset = CategoryVerificationParticipantOriginal()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="original", colour="blueviolet", symbol="+")
                )
            if participant_datasets in {ParticipantDataset.replication, ParticipantDataset.original_plus_replication}:
                participant_dataset = CategoryVerificationParticipantReplication()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
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
        participant_agreement(validation_run, participant_datasets, agreement_path)


def participant_agreement(validation_run: bool, participant_datasets: ParticipantDataset, agreement_path: Path):

    # Haven't done these yet!
    if not validation_run or  participant_datasets != ParticipantDataset.balanced:
        logger.warning("Skipping participant-agreement calculation for this dataset, not yet implemented!")
        return

    item_data = CategoryVerificationItemDataValidationBalanced()
    participant_data = CategoryVerificationParticipantBalancedValidation().data

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
    set_theme(style="darkgrid")

    g = jointplot(data=df, x=ColNames.ResponseAffirmativeProportion, y=MODEL_PEAK_ACTIVATION,
                  kind="reg", truncate=False,
                  marginal_kws={"kde": False})

    g.fig.savefig(str(Path(save_dir, f"{filename_prefix} model peak vs affirmative prop {filename_suffix}.png")), dpi=1200, bbox_inches='tight')
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
                        participant_plot_data.symbol, color=participant_plot_data.colour)
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
    ax.set_aspect('equal')
    pyplot.legend(legend_items)

    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.png"), dpi=1200 ,bbox_inches='tight')
    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}.svg"), dpi=1200 ,bbox_inches='tight')
    pyplot.close(fig)


@dataclass
class ArgSet:
    validation_run: bool
    participant_datasets: Optional[ParticipantDataset]
    items_matching_participant_dataset: ParticipantDataset

    exclude_repeated_items: bool = True
    restrict_to_answerable_items: bool = True
    use_assumed_object_label: bool = False

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
