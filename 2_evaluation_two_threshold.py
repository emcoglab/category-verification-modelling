#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Evaluating the combined model script for category verification task using the
two-threshold decision procedure.
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
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Dict, List

from numpy import inf
from numpy.random import seed
from pandas import DataFrame

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.ldm.utils.logging import print_progress
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import ColNames, CategoryVerificationParticipantOriginal, \
    CategoryObjectPair
from framework.evaluation.decision import performance_for_two_thresholds
from framework.evaluation.figures import save_heatmap
from framework.evaluation.load import load_model_output_from_dir

_logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# Paths
ROOT_INPUT_DIR = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")

# Shared
_n_thresholds = 10
THRESHOLDS = [i / _n_thresholds for i in range(_n_thresholds + 1)]  # linspace was causing weird float rounding errors


def main(spec: CategoryVerificationJobSpec, spec_filename: str, exclude_repeated_items: bool, restrict_to_answerable_items: bool, overwrite: bool):
    """
    :param: exclude_repeated_items:
        If yes, where a category and item are identical (GRASSHOPPER - grasshopper) or the latter includes the former
        (CUP - paper cup), the items are excluded from further analysis.
    """

    _logger.info("")
    _logger.info(f"Spec: {spec_filename}")

    model_output_dir = Path(ROOT_INPUT_DIR, spec.output_location_relative())
    if not model_output_dir.exists():
        _logger.warning(f"Model output not found for v{VERSION} in directory {model_output_dir.as_posix()}")
        return
    if not Path(model_output_dir, " MODEL RUN COMPLETE").exists():
        _logger.info(f"Incomplete model run found in {model_output_dir.as_posix()}")
        return
    save_dir = Path(model_output_dir, " evaluation")
    if save_dir.exists() and not overwrite:
        _logger.info(f"Evaluation complete for {save_dir.as_posix()}")
        return
    save_dir.mkdir(parents=False, exist_ok=True)

    try:
        all_model_data: Dict[CategoryObjectPair, DataFrame] = load_model_output_from_dir(model_output_dir, exclude_repeated_items=exclude_repeated_items)
    except FileNotFoundError:
        _logger.warning(f"No model data in {model_output_dir.as_posix()}")
        return

    correct_rates = []
    dprimes = []
    criteria = []
    threshold_i = 0
    for decision_threshold_no in THRESHOLDS:
        for decision_threshold_yes in THRESHOLDS:
            if decision_threshold_no >= decision_threshold_yes:
                continue
            threshold_i += 1

            correct_rate, dprime, criterion = performance_for_two_thresholds(
                all_model_data=all_model_data,
                restrict_to_answerable_items=restrict_to_answerable_items,
                decision_threshold_yes=decision_threshold_yes,
                decision_threshold_no=decision_threshold_no,
                loglinear=True,
                spec=spec, save_dir=Path(save_dir, "hitrates by threshold"))
            correct_rates.append((decision_threshold_no, decision_threshold_yes, correct_rate))
            dprimes.append((decision_threshold_no, decision_threshold_yes, dprime))
            criteria.append((decision_threshold_no, decision_threshold_yes, criterion))

            print_progress(threshold_i, len(THRESHOLDS) * (len(THRESHOLDS) - 1) / 2, prefix="Running Yes/No thresholds: ", bar_length=50)

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'

    items_subset: List[CategoryObjectPair] = list(all_model_data.keys()) if restrict_to_answerable_items else None

    participant_dprime_mean    = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.DPrime_loglinear].mean()
    participant_dprime_sd      = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.DPrime_loglinear].std()
    participant_criterion_mean = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.Criterion_loglinear].mean()
    participant_criterion_sd   = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.Criterion_loglinear].std()

    # Save overall dprimes
    dprimes_df = DataFrame.from_records(
        dprimes,
        columns=["Decision threshold (no)", "Decision threshold (yes)", ColNames.DPrime_loglinear])
    with Path(save_dir, f"{filename_prefix} dprimes.csv").open("w") as f:
        dprimes_df.to_csv(f, header=True, index=False)
    dprimes_df[f"{ColNames.DPrime_loglinear} absolute difference"] = abs(participant_dprime_mean - dprimes_df[ColNames.DPrime_loglinear])
    save_heatmap(dprimes_df, Path(save_dir, f"{filename_prefix} dprimes.png"), value_col=ColNames.DPrime_loglinear, vlims=(None, None))
    save_heatmap(dprimes_df, Path(save_dir, f"{filename_prefix} dprimes difference.png"), value_col=f"{ColNames.DPrime_loglinear} absolute difference", vlims=(0, None))

    # Save overall criteria
    criteria_df = DataFrame.from_records(
        criteria,
        columns=["Decision threshold (no)", "Decision threshold (yes)", ColNames.Criterion_loglinear])
    with Path(save_dir, f"{filename_prefix} criteria.csv").open("w") as f:
        criteria_df.to_csv(f, header=True, index=False)
    # Difference to subject-average criterion
    criteria_df[f"{ColNames.Criterion_loglinear} absolute difference"] = abs(participant_criterion_mean - criteria_df[ColNames.Criterion_loglinear])
    save_heatmap(criteria_df, Path(save_dir, f"{filename_prefix} criteria.png"), value_col=ColNames.Criterion_loglinear, vlims=(None, None))
    save_heatmap(criteria_df, Path(save_dir, f"{filename_prefix} criteria difference.png"), value_col=f"{ColNames.Criterion_loglinear} absolute difference", vlims=(0, None))

    # Find max suitable dprime
    max_dprime = -inf
    max_no, max_yes = None, None
    correct_rate_for_max_dprime, criterion_for_max_dprime = None, None
    max_correct_rate = -inf
    for (no_th, yes_th, dprime), (_, _, criterion), (_, _, correct_rate) in zip(dprimes, criteria, correct_rates):
        if participant_criterion_mean - participant_criterion_sd <= criterion <= participant_criterion_mean + participant_criterion_sd:
            if dprime > max_dprime:
                max_dprime = dprime
                # Remember other params for the best dprime
                max_no, max_yes = no_th, yes_th
                correct_rate_for_max_dprime, criterion_for_max_dprime = correct_rate, criterion
            if correct_rate > max_correct_rate:
                max_correct_rate = correct_rate
    _logger.info(f"Participant dprime mean (SD): {participant_dprime_mean} ({participant_dprime_sd})")
    _logger.info(f"Participant criterion mean (SD): {participant_criterion_mean} ({participant_criterion_sd})")
    _logger.info(f"Best dprime for which criterion is within 1SD of participant mean: {max_dprime} (no={max_no}, yes={max_yes})")
    _logger.info(f'Criterion for this dprime: {criterion_for_max_dprime}')
    _logger.info(f"Correct-rate for this dprime: {correct_rate_for_max_dprime}")
    _logger.info(f"Best correct rate overall: {max_correct_rate}")


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
        "2022-01-24 More variations on the current favourite.yaml",
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
             overwrite=True)

    _logger.info("Done!")
