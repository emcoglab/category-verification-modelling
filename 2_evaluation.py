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

import sys
from copy import deepcopy
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Optional

from matplotlib import pyplot
from numpy import savetxt, array
from pandas import DataFrame

from framework.cli.job import InteractiveCombinedJobSpec
from framework.evaluation.column_names import TTFA, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f, PRODUCTION_PROPORTION, \
    RANKED_PRODUCTION_FREQUENCY, ROUNDED_MEAN_RANK, COMPONENT

from framework.data.category_verification import CategoryVerificationOriginal, CategoryVerificationReplication

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# arg choices: dataset
ARG_DATASET_TRAIN = "train"
ARG_DATASET_TEST  = "test"

# Paths
root_input_dir = Path("/Volumes/Big Data/spreading activation model/Model output/Category verification")


def prepare_category_verification_data(model_type: ModelType, dataset: str) -> DataFrame:

    # Select correct dataset
    # TODO: not sure we're actually using the differences here, if the stimuli are the same
    if dataset == ARG_DATASET_TRAIN:
        cv_data = CategoryVerificationOriginal()
    elif dataset == ARG_DATASET_TEST:
        cv_data = CategoryVerificationReplication()
    else:
        raise ValueError()

    raise NotImplementedError()


def prepare_main_dataframe(spec: InteractiveCombinedJobSpec, filter_events: Optional[str], accessible_set_hits: bool, dataset: str) -> DataFrame:

    main_data: DataFrame = prepare_category_verification_data(ModelType.combined_interactive, dataset)

    # TODO


    return main_data


def main(spec: InteractiveCombinedJobSpec, manual_cut_off: Optional[int] = None, filter_events: Optional[str] = None,
         accessible_set_hits: bool = False):

    main_data = prepare_main_dataframe(spec=spec, filter_events=filter_events, accessible_set_hits=accessible_set_hits)

    model_output_dir = Path(root_input_dir, spec.output_location_relative())
    if filter_events is not None:
        model_output_dir = Path(model_output_dir.parent, model_output_dir.name + f" only {filter_events}")
    if accessible_set_hits:
        evaluation_output_dir = Path(model_output_dir, " Evaluation (accessible set hits)")
    else:
        evaluation_output_dir = Path(model_output_dir, " Evaluation")
    evaluation_output_dir.mkdir(exist_ok=True)

    if manual_cut_off is None:
        explore_ttfa_cutoffs(main_data, evaluation_output_dir)
    else:
        fit_data_at_cutoff(main_data, evaluation_output_dir, manual_cut_off)

    # Save final main dataframe
    with open(Path(evaluation_output_dir, f"main model data.csv"), mode="w", encoding="utf-8") as main_data_output_file:
        main_data[[
            # Select only relevant columns for output
            CPColNames.Category,
            CPColNames.Response,
            CPColNames.CategorySensorimotor,
            CPColNames.ResponseSensorimotor,
            CPColNames.ProductionFrequency,
            CPColNames.MeanRank,
            CPColNames.FirstRankFrequency,
            CPColNames.MeanRT,
            CPColNames.MeanZRT,
            PRODUCTION_PROPORTION,
            RANKED_PRODUCTION_FREQUENCY,
            ROUNDED_MEAN_RANK,
            TTFA,
            COMPONENT,
        ]].to_csv(main_data_output_file, index=False)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    loaded_specs = InteractiveCombinedJobSpec.load_multiple(Path(Path(__file__).parent,
                                                                 "job_specifications",
                                                                 "2021-05-06 interactive testing batch.yaml"))
    systematic_cca_test = True

    if systematic_cca_test:
        ccas = [0.0, 0.5, 1.0]
        specs = []
        s: InteractiveCombinedJobSpec
        for s in loaded_specs:
            for cca in ccas:
                spec = deepcopy(s)
                spec.cross_component_attenuation = cca
                specs.append(spec)
    else:
        specs = loaded_specs

    for spec in specs:
        main(spec=spec, filter_events="accessible_set", accessible_set_hits=False)

    logger.info("Done!")

