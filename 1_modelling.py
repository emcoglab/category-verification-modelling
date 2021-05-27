#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Interactive combined model script for category verification task.
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

from enum import Enum
from sys import argv
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from numpy import nan
from pandas import DataFrame

from framework.cognitive_model.components import FULL_ACTIVATION
from framework.data.category_verification import CategoryVerificationOriginal, CategoryVerificationReplication
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.basic_types import ActivationValue, Length
from framework.cognitive_model.combined_cognitive_model import InteractiveCombinedCognitiveModel
from framework.cognitive_model.linguistic_components import LinguisticComponent
from framework.cognitive_model.sensorimotor_components import SensorimotorComponent
from framework.cognitive_model.attenuation_statistic import AttenuationStatistic
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.preferences.preferences import Preferences
from framework.cli.job import CategoryVerificationJobSpec, LinguisticPropagationJobSpec, SensorimotorPropagationJobSpec

# arg choices: filter_events
ARG_ACCESSIBLE_SET = "accessible_set"
ARG_BUFFER         = "buffer"

# arg choices: dataset
ARG_DATASET_TRAIN = "train"
ARG_DATASET_TEST  = "test"


class Decision(Enum):
    yes = 1
    no = 0
    undecided = -1


def check_for_decision(job_spec, model, object_label):
    # TODO: which component to track object activation in?
    object_activation = model.linguistic_component.propagator.activation_of_item_with_label(object_label)
    if object_activation < job_spec.decision_threshold_no:
        return Decision.no
    if object_activation >= job_spec.decision_threshold_yes:
        return Decision.yes
    return Decision.undecided


def main(job_spec: CategoryVerificationJobSpec, use_prepruned: bool, filter_events: Optional[str], dataset: str):

    # Validate spec
    assert job_spec.soa_ticks <= job_spec.run_for_ticks

    # Set up output directories
    response_dir: Path = Path(Preferences.output_dir, "Category verification", job_spec.output_location_relative())
    if filter_events is not None:
        response_dir = Path(response_dir.parent, response_dir.name + f" only {filter_events}")
    if not response_dir.is_dir():
        logger.warning(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)

    # Set up model
    model = InteractiveCombinedCognitiveModel(
        sensorimotor_component=(job_spec.sensorimotor_spec.to_component_prepruned(SensorimotorComponent)
                                if use_prepruned
                                else job_spec.sensorimotor_spec.to_component(SensorimotorComponent)),
        linguistic_component=job_spec.linguistic_spec.to_component(LinguisticComponent),
        lc_to_smc_delay=job_spec.lc_to_smc_delay,
        smc_to_lc_delay=job_spec.smc_to_lc_delay,
        lc_to_smc_threshold=job_spec.lc_to_smc_threshold,
        smc_to_lc_threshold=job_spec.smc_to_lc_threshold,
        cross_component_attenuation=job_spec.cross_component_attenuation,
        buffer_threshold=job_spec.buffer_threshold,
        buffer_capacity_linguistic_items=job_spec.buffer_capacity_linguistic_items,
        buffer_capacity_sensorimotor_items=job_spec.buffer_capacity_sensorimotor_items,
    )
    job_spec.save(in_location=response_dir)
    model.mapping.save_to(directory=response_dir)

    # Select correct dataset
    # TODO: not sure we're actually using the differences here, if the stimuli are the same
    if dataset == ARG_DATASET_TRAIN:
        cv_data = CategoryVerificationOriginal()
    elif dataset == ARG_DATASET_TEST:
        cv_data = CategoryVerificationReplication()
    else:
        raise ValueError()

    object_activation_increment: ActivationValue = job_spec.object_activation / job_spec.incremental_activation_duration
    for category_label, object_label in cv_data.category_object_pairs():
        model.reset()

        # (1) Activate the initial category label
        # TODO: are any multi-word?
        model.linguistic_component.propagator.activate_item_with_label(category_label, activation=FULL_ACTIVATION)

        # Start the clock
        for _ in range(0, job_spec.soa_ticks):
            logger.info(f"Clock = {model.clock}")

            model.tick()

        # Once we reach the SOA time, begin checking for a decision
        for _ in range(job_spec.soa_ticks, job_spec.run_for_ticks):

            # Apply incremental activation during the immediate post-SOA period
            if model.clock < job_spec.soa_ticks + job_spec.incremental_activation_duration:
                # TODO: attenuate this by prevalence
                model.sensorimotor_component.propagator.activate_item_with_label(object_label, activation=object_activation_increment)
                model.linguistic_component.propagator.activate_item_with_label(object_label, activation=object_activation_increment)

            # Advance the model
            model.tick()

            # Once we've reached point (3), we start checking for a decision
            if model.clock < job_spec.soa_ticks:
                continue

            # TODO !!!!!!!!!!
            #  The time we'll save by stopping when we have a decision, we'll definitely waste by having to rerun the
            #  whole thing for a new pair of thresholds.  We should just record the activation and do the deciding.
            decision = check_for_decision(job_spec, model, object_label)
            if decision == Decision.undecided:
                continue





if __name__ == '__main__':
    logger.info("Running %s" % " ".join(argv))

    parser = ArgumentParser(description="Run interactive combined model.")

    parser.add_argument("--linguistic_accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_accessible_set_capacity", required=False, type=int)
    parser.add_argument("--linguistic_use_activation_cap", action="store_true")
    parser.add_argument("--linguistic_corpus_name", required=True, type=str)
    parser.add_argument("--linguistic_firing_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_impulse_pruning_threshold", required=True, type=ActivationValue)
    parser.add_argument("--linguistic_length_factor", required=True, type=int)
    parser.add_argument("--linguistic_model_name", required=True, type=str)
    parser.add_argument("--linguistic_node_decay_factor", required=True, type=float)
    parser.add_argument("--linguistic_radius", required=True, type=int)
    parser.add_argument("--linguistic_edge_decay_sd_factor", required=True, type=float)
    parser.add_argument("--linguistic_words", type=int, required=True)

    parser.add_argument("--sensorimotor_accessible_set_threshold", required=True, type=ActivationValue)
    parser.add_argument("--sensorimotor_accessible_set_capacity", required=False, type=int)
    parser.add_argument("--sensorimotor_use_activation_cap", action="store_true")
    parser.add_argument("--sensorimotor_distance_type", required=True, type=str)
    parser.add_argument("--sensorimotor_length_factor", required=True, type=Length)
    parser.add_argument("--sensorimotor_node_decay_median", required=True, type=float)
    parser.add_argument("--sensorimotor_node_decay_sigma", required=True, type=float)
    parser.add_argument("--sensorimotor_max_sphere_radius", required=True, type=float)
    parser.add_argument("--sensorimotor_use_prepruned", action="store_true")
    parser.add_argument("--sensorimotor_attenuation", required=True, type=str, choices=[n.name for n in AttenuationStatistic])
    # We have to add this argument to make the interface compatible, but we always use the BrEng translation
    parser.add_argument("--sensorimotor_use_breng_translation", action="store_true")

    parser.add_argument("--buffer_threshold", required=True, type=ActivationValue)
    parser.add_argument("--buffer_capacity_linguistic_items", required=True, type=int)
    parser.add_argument("--buffer_capacity_sensorimotor_items", required=True, type=int)
    parser.add_argument("--lc_to_smc_delay", required=True, type=int)
    parser.add_argument("--smc_to_lc_delay", required=True, type=int)
    parser.add_argument("--lc_to_smc_threshold", required=True, type=ActivationValue)
    parser.add_argument("--smc_to_lc_threshold", required=True, type=ActivationValue)
    parser.add_argument("--cross_component_attenuation", required=True, type=float)
    parser.add_argument("--bailout", required=False, default=0, type=int)
    parser.add_argument("--run_for_ticks", required=True, type=int)

    parser.add_argument("--filter_events", type=str, choices=[ARG_BUFFER, ARG_ACCESSIBLE_SET], default=None)

    parser.add_argument("--soa", type=int, required=True)
    parser.add_argument("--object_activation", type=ActivationValue, required=True)
    parser.add_argument("--object_activation_duration", type=int, required=True)
    parser.add_argument("--yes_threshold", type=ActivationValue, required=True)
    parser.add_argument("--no_threshold", type=ActivationValue, required=True)

    # Additional args
    parser.add_argument("--dataset", type=str, choices=[ARG_DATASET_TRAIN, ARG_DATASET_TEST], required=True)

    args = parser.parse_args()

    if not args.sensorimotor_use_breng_translation:
        logger.warning("BrEng translation will always be used in the interactive model.")

    main(
        job_spec=CategoryVerificationJobSpec(
            linguistic_spec=LinguisticPropagationJobSpec(
                accessible_set_threshold=args.linguistic_accessible_set_threshold,
                accessible_set_capacity=args.linguistic_accessible_set_capacity,
                use_activation_cap=args.linguistic_use_activation_cap,
                corpus_name=args.linguistic_corpus_name,
                firing_threshold=args.linguistic_firing_threshold,
                impulse_pruning_threshold=args.linguistic_impulse_pruning_threshold,
                length_factor=args.linguistic_length_factor,
                model_name=args.linguistic_model_name,
                node_decay_factor=args.linguistic_node_decay_factor,
                model_radius=args.linguistic_radius,
                edge_decay_sd=args.linguistic_edge_decay_sd_factor,
                n_words=args.linguistic_words,
                pruning=None,
                pruning_type=None,
                bailout=args.bailout,
                run_for_ticks=args.run_for_ticks,
            ),
            sensorimotor_spec=SensorimotorPropagationJobSpec(
                accessible_set_threshold=args.sensorimotor_accessible_set_threshold,
                accessible_set_capacity=args.sensorimotor_accessible_set_capacity,
                use_activation_cap=args.sensorimotor_use_activation_cap,
                distance_type=DistanceType.from_name(args.sensorimotor_distance_type),
                length_factor=args.sensorimotor_length_factor,
                node_decay_median=args.sensorimotor_node_decay_median,
                node_decay_sigma=args.sensorimotor_node_decay_sigma,
                attenuation_statistic=AttenuationStatistic.from_slug(args.sensorimotor_attenuation),
                max_radius=args.sensorimotor_max_sphere_radius,
                use_breng_translation=True,
                bailout=args.bailout,
                run_for_ticks=args.run_for_ticks,
            ),
            buffer_threshold=args.buffer_threshold,
            buffer_capacity_linguistic_items=args.buffer_capacity_linguistic_items,
            buffer_capacity_sensorimotor_items=args.buffer_capacity_sensorimotor_items,
            cross_component_attenuation=args.cross_component_attenuation,
            lc_to_smc_delay=args.lc_to_smc_delay,
            smc_to_lc_delay=args.smc_to_lc_delay,
            lc_to_smc_threshold=args.lc_to_smc_threshold,
            smc_to_lc_threshold=args.smc_to_lc_threshold,
            run_for_ticks=args.run_for_ticks,
            bailout=args.bailout,
            soa_ticks=args.soa,
            object_activation=args.object_activation,
            incremental_activation_duration=args.object_activation_duration,
            decision_threshold_yes=args.yes_threshold,
            decision_threshold_no=args.no_threshold,
        ),
        use_prepruned=args.sensorimotor_use_prepruned,
        filter_events=args.filter_events,
        dataset=args.dataset,
    )

    logger.info("Done!")
