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
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot
from numpy import trapz
from numpy.random import seed
from pandas import DataFrame, Series

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import ColNames, CategoryObjectPair, Filter, \
    CategoryVerificationParticipantOriginal, CategoryVerificationParticipantReplication, \
    CategoryVerificationItemData, CategoryVerificationItemDataBlockedValidation
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import OBJECT_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f
from framework.evaluation.load import load_model_output_from_dir

_logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")

# Shared
_n_threshold_steps = 10
THRESHOLDS = [i / _n_threshold_steps for i in range(_n_threshold_steps + 1)]  # linspace was causing weird float rounding errors

# Additional col names
MODEL_GUESS = "Model guessd"
MODEL_PEAK_ACTIVATION = "Model peak post-SOA activation"


def main(spec: CategoryVerificationJobSpec, spec_filename: str, exclude_repeated_items: bool,
         restrict_to_answerable_items: bool, use_assumed_object_label: bool, validation_run: bool,
         participant_original_dataset: bool, participant_replication_dataset: bool,
         overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    _logger.info("")
    _logger.info(f"Spec: {spec_filename}")

    # Determine directory paths with optional tests for early exit
    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    if validation_run:
        model_output_dir = Path(model_output_dir, "validation")
    if not model_output_dir.exists():
        _logger.warning(f"Model output not found for v{VERSION} in directory {model_output_dir.as_posix()}")
        return
    if not Path(model_output_dir, " MODEL RUN COMPLETE").exists():
        _logger.warning(f"Incomplete model run found in {model_output_dir.as_posix()}")
        return
    save_dir = Path(model_output_dir, " evaluation")
    if save_dir.exists() and not overwrite:
        _logger.info(f"Evaluation complete for {save_dir.as_posix()}")
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

    def get_peak_activation(row) -> Optional[float]:
        item_col = ColNames.ImageLabelAssumed if use_assumed_object_label else ColNames.ImageObject
        cop = CategoryObjectPair(category_label=row[ColNames.CategoryLabel], object_label=row[item_col])
        try:
            model_activations_df: DataFrame = model_data[cop]
        except KeyError:
            return None
        # The decision rests on the peak activation over object labels over both components, so we can just take the max
        # of all of them
        object_label_linguistic, object_label_sensorimotor = substitutions_for(cop.object_label)
        object_label_linguistic_multiword_parts: List[str] = modified_word_tokenize(object_label_linguistic)
        # We are only interested in the activation after ths SOA
        post_soa_df = model_activations_df.loc[spec.soa_ticks+1:spec.run_for_ticks]
        peak_activation_linguistic = post_soa_df[OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)].max()
        peak_activation_sensorimotor = max(
            post_soa_df[OBJECT_ACTIVATION_LINGUISTIC_f.format(part)].max()
            for part in object_label_linguistic_multiword_parts
        )
        return max(peak_activation_linguistic, peak_activation_sensorimotor)

    for cv_filter in filters:

        # apply filters
        if validation_run:
            filtered_df = CategoryVerificationItemDataBlockedValidation().dataframe_filtered(cv_filter)
        else:
            filtered_df = CategoryVerificationItemData().dataframe_filtered(cv_filter)
        filtered_df[MODEL_PEAK_ACTIVATION] = filtered_df.apply(get_peak_activation, axis=1)

        if restrict_to_answerable_items:
            filtered_df.dropna(subset=[MODEL_PEAK_ACTIVATION], inplace=True)

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
        if participant_original_dataset and participant_replication_dataset:
            filename_prefix += " all participants"
        elif participant_original_dataset:
            filename_prefix += " original participants"
        elif participant_replication_dataset:
            filename_prefix += " replication participants"
        filename_suffix = cv_filter.name

        participant_plot_datasets = []
        if validation_run:
            # Don't have participant data for this
            pass

        else:
            # Participant hitrates
            if participant_original_dataset:
                participant_dataset = CategoryVerificationParticipantOriginal()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemData.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="original", colour="g")
                )
            if participant_replication_dataset:
                participant_dataset = CategoryVerificationParticipantReplication()
                participant_summary_df = participant_dataset.participant_summary_dataframe(
                    use_item_subset=CategoryVerificationItemData.list_category_object_pairs_from_dataframe(
                        filtered_df, use_assumed_object_label=use_assumed_object_label))
                participant_plot_datasets.append(
                    ParticipantPlotData(hit_rates=participant_summary_df[ColNames.HitRate],
                                        fa_rates=participant_summary_df[ColNames.FalseAlarmRate],
                                        dataset_name="replication", colour="c")
                )

        plot_roc(model_hit_rates, model_false_alarm_rates,
                 participant_plot_datasets,
                 filename_prefix, filename_suffix, save_dir)

        with Path(save_dir, f"{filename_prefix} data {filename_suffix}.csv") as f:
            filtered_df.to_csv(f, index=False)


@dataclass
class ParticipantPlotData:
    hit_rates: Series
    fa_rates: Series
    dataset_name: str
    colour: str


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
             filename_prefix, filename_suffix, save_dir):

    fig, ax = pyplot.subplots()

    # AUC
    auc = trapz(list(reversed(model_hit_rates)), list(reversed(model_fa_rates)))

    # Identity line
    pyplot.plot([0, 1], [0, 1], "r--")
    # Model
    pyplot.plot(model_fa_rates, model_hit_rates, "b-")

    legend_items = ["Random classifier", "Model"]
    participant_aucs = []
    for participant_plot_data in participant_plot_datasets:
        # Participant points
        pyplot.plot(participant_plot_data.fa_rates, participant_plot_data.hit_rates, f"{participant_plot_data.colour}+")
        # Participant mean spline interpolation
        # pyplot.plot(participant_interpolated_x, participant_interpolated_y, "g--")
        # Participant linearly interpolated areas
        for participant_fa, participant_hit in zip(participant_plot_data.fa_rates, participant_plot_data.hit_rates):
            px = [0, participant_fa, 1]
            py = [0, participant_hit, 1]
            pyplot.fill_between(px, py, color=(0, 0, 0, 0.02), label='_nolegend_')
            participant_aucs.append(trapz(py, px))

        legend_items.append(f"Participants ({participant_plot_data.dataset_name} dataset)")

    if participant_plot_datasets:
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

    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC {filename_suffix}"))
    pyplot.close(fig)


# noinspection DuplicatedCode
if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    _logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    loaded_specs = []
    for sfn in [
        # "2021-08-16 educated guesses.yaml",
        # "2021-07-15 40k different decay.yaml",
        # "2021-06-25 search for more sensible parameters.yaml",
        # "2021-09-07 Finer search around a good model.yaml",
        # "2021-09-14 Finer search around another good model.yaml",
        # "2022-01-24 More variations on the current favourite.yaml",
        # "2022-05-06 A slightly better one-threshold model.yaml",
        "2022-07-15 good roc-auc candidate.yaml",
        # "2022-07-25 slower linguistic decay experiment.yaml",
        # "2022-08-01 varying soa.yaml",
    ]:
        loaded_specs.extend([(s, sfn, i) for i, s in enumerate(CategoryVerificationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications", sfn)))])

    systematic_cca_test = False
    if systematic_cca_test:
        ccas = [1.0, 0.5, 0.0]
        specs = []
        s: CategoryVerificationJobSpec
        for s, sfn, i in loaded_specs:
            for cca in ccas:
                spec = deepcopy(s)
                spec.cross_component_attenuation = cca
                specs.append((spec, sfn, i))
    else:
        specs = loaded_specs

    for j, (spec, sfn, i) in enumerate(specs, start=1):
        _logger.info(f"Evaluating model {j} of {len(specs)}")
        main(spec=spec,
             spec_filename=f"{sfn} [{i}]",
             exclude_repeated_items=True,
             restrict_to_answerable_items=True,
             use_assumed_object_label=False,
             validation_run=True,
             participant_original_dataset=False,
             participant_replication_dataset=False,
             overwrite=True)

    _logger.info("Done!")
