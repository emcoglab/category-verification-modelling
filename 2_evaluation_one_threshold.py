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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from matplotlib import pyplot
from numpy import trapz, isnan, nan
from numpy.random import seed
from pandas import DataFrame, Series
from seaborn import jointplot
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


def main(spec: CategoryVerificationJobSpec, spec_filename: str, exclude_repeated_items: bool,
         restrict_to_answerable_items: bool, validation_run: bool,
         participant_datasets: Optional[ParticipantDataset],
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
        trial_type_filter = None
    else:
        trial_type_filter = Filter.new_trial_type_filter({('test', True), ('filler', False)})
    if exclude_repeated_items:
        repeated_items_filter = Filter.new_repeated_item_filter(tokeniser=modified_word_tokenize)
    else:
        repeated_items_filter = None

    filter_sets: Dict[str, List[Filter | None]] = {
        "both": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["both"]),
            trial_type_filter,
            repeated_items_filter
        ],
        "superordinate": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["superordinate"]),
            trial_type_filter,
            repeated_items_filter
        ],
        "basic": [
            Filter.new_category_taxonomic_level_filter(allowed_levels=["basic"]),
            trial_type_filter,
            repeated_items_filter
        ]
    }

    # When validating, we can break down by category domains
    if validation_run:
        new_filter_sets: Dict[str, List[Filter]] = dict()
        for name, filter_set in filter_sets.items():
            for category_domain in ["natural", "artefact"]:
                if category_domain == "natural" and name == "superordinate":
                    # There are no items here
                    # TODO: this convolution seems a little silly
                    continue
                new_filter_sets[f"{category_domain} {name}"] = filter_set + [Filter.new_category_domain_filter([category_domain])]
        filter_sets = new_filter_sets

    # Add model peak activations
    model_data: Dict[CategoryObjectPair, DataFrame] = load_model_output_from_dir(model_output_dir, validation=validation_run, for_participant_dataset=participant_datasets)

    def get_peak_activation(row, *, allow_missing_objects: bool) -> float:
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

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'
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

    for filter_set_name, filter_set in filter_sets:

        # apply filters
        if validation_run:
            if participant_datasets == ParticipantDataset.validation:
                filtered_df = Filter.apply_filters(filters=filter_set,
                                                   to_df=CategoryVerificationItemDataBlockedValidation().data)
            elif participant_datasets == ParticipantDataset.balanced:
                filtered_df = Filter.apply_filters(filters=filter_set,
                                                   to_df=CategoryVerificationItemDataValidationBalanced().data)
            else:
                raise NotImplementedError()
        else:
            if participant_datasets == ParticipantDataset.original:
                filtered_df = Filter.apply_filters(filters=filter_set,
                                                   to_df=CategoryVerificationItemDataOriginal().data)
            elif participant_datasets == ParticipantDataset.replication:
                filtered_df = Filter.apply_filters(filters=filter_set,
                                                   to_df=CategoryVerificationItemDataReplication().data)
            elif participant_datasets == ParticipantDataset.original_plus_replication:
                filtered_df = Filter.apply_filters(filters=filter_set,
                                                   to_df=CategoryVerificationItemDataOriginal().data)
                logger.warning(
                    "Participant-related values not yet correct when using all participants, these will be omitted.")
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
        filename_suffix = filter_set_name

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
                                        dataset_name="validation", colour="forestgreen")
                )
            if participant_datasets in {ParticipantDataset.balanced, ParticipantDataset.validation_plus_balanced}:
                participant_dataset = CategoryVerificationParticipantBalancedValidation()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataBlockedValidation.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="balanced", colour="lightseagreen")
                )

        else:
            if participant_datasets in {ParticipantDataset.original, ParticipantDataset.original_plus_replication}:
                participant_dataset = CategoryVerificationParticipantOriginal()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(
                        filtered_df))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="original", colour="blueviolet")
                )
            if participant_datasets in {ParticipantDataset.replication, ParticipantDataset.original_plus_replication}:
                participant_dataset = CategoryVerificationParticipantReplication()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemDataOriginal.list_category_object_pairs_from_dataframe(filtered_df))
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

        plot_peak_activation_vs_affirmative_proportion(
            filtered_df,
            filename_prefix, filename_suffix, save_dir,
        )

        with Path(save_dir, f"{filename_prefix} data {filename_suffix}.csv") as f:
            filtered_df.to_csv(f, index=False)

    agreement_path: Path = Path(save_dir, f"{filename_prefix} agreement.csv")
    participant_agreement(validation_run, participant_datasets, agreement_path)


def participant_agreement(validation_run: bool, participant_datasets: ParticipantDataset, agreement_path: Path):

    # Haven't done these yet!
    if not validation_run:
        raise NotImplementedError()
    if participant_datasets != ParticipantDataset.balanced:
        raise NotImplementedError()

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
    from seaborn import set_theme
    set_theme(style="darkgrid")  # Todo: globally

    g = jointplot(data=df, x=ColNames.ResponseAffirmativeProportion, y=MODEL_PEAK_ACTIVATION,
                  kind="reg", truncate=False,
                  marginal_kws={"kde": False})

    g.fig.savefig(str(Path(save_dir, f"{filename_prefix} model peak vs affirmative prop {filename_suffix}.png")))
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

    spec: CategoryVerificationJobSpec
    for j, (spec, sfn, i) in enumerate(loaded_specs, start=1):
        logger.info(f"Evaluating model {j} of {len(loaded_specs)}")
        for no_propagation in [False, True]:
            main(
                spec=spec,
                spec_filename=f"{sfn} [{i}]",
                exclude_repeated_items=True,
                restrict_to_answerable_items=True,
                validation_run=True,
                participant_datasets=ParticipantDataset.balanced,
                overwrite=True,
                no_propagation=no_propagation,
            )

    logger.info("Done!")
