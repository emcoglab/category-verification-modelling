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
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot
from numpy import linspace, mean, trapz
from numpy.random import seed
from pandas import DataFrame
from scipy import interpolate

from framework.cli.job import CategoryVerificationJobSpec
from framework.cognitive_model.ldm.utils.logging import print_progress
from framework.cognitive_model.version import VERSION
from framework.data.category_verification_data import ColNames, \
    CategoryVerificationParticipantOriginal, \
    CategoryObjectPair
from framework.evaluation.decision import performance_for_one_threshold
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

    hit_rates = []
    false_alarm_rates = []
    threshold_i = 0
    for decision_threshold in THRESHOLDS:
        threshold_i += 1

        hit_rate, fa_rate = performance_for_one_threshold(
            all_model_data=all_model_data,
            restrict_to_answerable_items=restrict_to_answerable_items,
            decision_threshold=decision_threshold,
            spec=spec, save_dir=Path(save_dir, "hitrates by threshold"),
            strict_inequality=True)
        hit_rates.append(hit_rate)
        false_alarm_rates.append(fa_rate)

        print_progress(threshold_i, len(THRESHOLDS), prefix="Running thresholds: ", bar_length=50)

    filename_prefix = 'excluding repeated items' if exclude_repeated_items else 'overall'

    items_subset: List[CategoryObjectPair] = list(all_model_data.keys()) if restrict_to_answerable_items else None

    participant_hit_rates = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.HitRate]
    participant_fa_rates = CategoryVerificationParticipantOriginal().participant_summary_dataframe(use_item_subset=items_subset)[ColNames.FalseAlarmRate]

    plot_roc(hit_rates, false_alarm_rates, participant_hit_rates, participant_fa_rates, filename_prefix, save_dir)


def plot_roc(model_hit_rates, model_fa_rates, participant_hit_rates, participant_fa_rates, filename_prefix, save_dir):

    fig, ax = pyplot.subplots()

    # AUC
    auc = trapz(list(reversed(model_hit_rates)), list(reversed(model_fa_rates)))

    # Interpolate
    anchor_points_x = [0, mean(participant_fa_rates), 1]
    anchor_points_y = [0, mean(participant_hit_rates), 1]
    participant_interpolated_x = linspace(0, 1, len(THRESHOLDS), endpoint=True)
    participant_interpolated_y = interpolate.pchip_interpolate(anchor_points_x, anchor_points_y, participant_interpolated_x)

    # Identity line
    pyplot.plot([0, 1], [0, 1], "r--")
    # Model
    pyplot.plot(model_fa_rates, model_hit_rates, "b-")
    # Participant points
    pyplot.plot(participant_fa_rates, participant_hit_rates, "g+")
    # Participant interpolation
    pyplot.plot(participant_interpolated_x, participant_interpolated_y, "g--")

    # Style graph
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"ROC curve (model AUC = {auc:.2})")
    pyplot.legend(["Random classifier", "Model", "Participants", "Participant interpolation"])

    pyplot.savefig(Path(save_dir, f"{filename_prefix} ROC"))
    pyplot.close(fig)


# noinspection DuplicatedCode
if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    _logger.info("Running %s" % " ".join(sys.argv))

    seed(1)  # Reproducible results

    loaded_specs = []
    for sfn in [
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
