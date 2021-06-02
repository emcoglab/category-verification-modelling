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

from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from typing import Optional

from pandas import DataFrame

from framework.cli.job import CategoryVerificationJobSpec, LinguisticPropagationJobSpec, SensorimotorPropagationJobSpec
from framework.cognitive_model.attenuation_statistic import AttenuationStatistic
from framework.cognitive_model.basic_types import ActivationValue, Length
from framework.cognitive_model.combined_cognitive_model import InteractiveCombinedCognitiveModel
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.linguistic_components import LinguisticComponent
from framework.cognitive_model.preferences.preferences import Preferences
from framework.cognitive_model.sensorimotor_components import SensorimotorComponent
from framework.cognitive_model.sensorimotor_norms.exceptions import WordNotInNormsError
from framework.cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from framework.cognitive_model.utils.exceptions import ItemNotFoundError
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.utils.maths import scale_prevalence_01, prevalence_from_fraction_known
from framework.data.category_verification_data import CategoryVerificationItemData
from framework.evaluation.column_names import CLOCK, CATEGORY_ACTIVATION_LINGUISTIC_f, \
    CATEGORY_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f, OBJECT_ACTIVATION_SENSORIMOTOR_f
from framework.utils import decompose_multiword

# arg choices: filter_events
_ARG_ACCESSIBLE_SET = "accessible_set"
_ARG_BUFFER         = "buffer"

_sn = SensorimotorNorms(use_breng_translation=True)  # Always use the BrEng translation in the interactive model


def _get_activation_data(model, category_label, object_label) -> dict:
    """Gets all the relevant activation data in the form of a dictionary."""
    return {
        CLOCK: model.clock,
        **{
            CATEGORY_ACTIVATION_LINGUISTIC_f.format(part)
            : model.linguistic_component.propagator.activation_of_item_with_label(part)
            for part in decompose_multiword(category_label)
        },
        CATEGORY_ACTIVATION_SENSORIMOTOR_f.format(category_label)
        : model.sensorimotor_component.propagator.activation_of_item_with_label(category_label),
        **{
            OBJECT_ACTIVATION_LINGUISTIC_f.format(part)
            : model.linguistic_component.propagator.activation_of_item_with_label(part)
            for part in decompose_multiword(object_label)
        },
        OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label)
        : model.sensorimotor_component.propagator.activation_of_item_with_label(object_label)
    }


def main(job_spec: CategoryVerificationJobSpec, use_prepruned: bool, filter_events: Optional[str]):

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

    # Stimuli are the same for both datasets so it doesn't matter which we use here
    cv_item_data = CategoryVerificationItemData()

    activation_tracking_data = []

    object_activation_increment: ActivationValue = job_spec.object_activation / job_spec.incremental_activation_duration
    for category_label, object_label in cv_item_data.category_object_pairs():

        category_multiword_parts = decompose_multiword(category_label)
        object_multiword_parts = decompose_multiword(object_label)

        activation_tracking_path = Path(response_dir, f"{category_label}-{object_label}.csv")

        model.reset()

        object_prevalence: float
        try:
            object_prevalence = scale_prevalence_01(prevalence_from_fraction_known(_sn.fraction_known(object_label)))
        except WordNotInNormsError:
            # In case the word isn't in the norms, make that known, but fall back to full prevalence
            logger.warning(f"Could not look up prevalence as {object_label} is not in the sensorimotor norms")
            object_prevalence = 1.0

        # (1) Activate the initial category label
        try:
            model.linguistic_component.propagator.activate_items_with_labels(
                labels=category_multiword_parts,
                activation=FULL_ACTIVATION / len(category_multiword_parts))
        except ItemNotFoundError as e:
            logger.error(f"Missing sensorimotor item: {object_label}")
            # raise e
            continue  # TODO: temporarily we don't raise, we just note the problem. Later we can decide what to do.
        # Start the clock
        for _ in range(0, job_spec.run_for_ticks):
            logger.info(f"Clock = {model.clock}")

            # Apply incremental activation during the immediate post-SOA period
            if job_spec.soa_ticks <= model.clock < job_spec.soa_ticks + job_spec.incremental_activation_duration:
                # Activate sensorimotor item directly
                try:
                    model.sensorimotor_component.propagator.activate_item_with_label(
                        label=object_label,
                        activation=object_activation_increment)
                except ItemNotFoundError as e:
                    logger.error(f"Missing sensorimotor item: {object_label}")
                    # raise e
                    pass  # TODO: decide what to do here too
                # Activate linguistic items separately
                try:
                    model.linguistic_component.propagator.activate_items_with_labels(
                        labels=object_multiword_parts,
                        # Attenuate linguistic activation by object label prevalence
                        activation=object_activation_increment * object_prevalence / len(object_multiword_parts))
                except ItemNotFoundError as e:
                    logger.error(f"Missing linguistic items: {object_multiword_parts}")
                    # raise e
                    pass  # TODO: decide what to do here too

            # Advance the model
            model.tick()

            # Record the relevant activations
            # There are a variable number of columns, depending on whether the items contain multiple words or not.
            # Therefore we record it in a list[dict] and build it into a DataFrame later for saving.
            activation_tracking_data.append(_get_activation_data(model, category_label, object_label))

        with activation_tracking_path.open("w") as file:
            DataFrame(activation_tracking_data).to_csv(file, index=False)


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

    parser.add_argument("--filter_events", type=str, choices=[_ARG_BUFFER, _ARG_ACCESSIBLE_SET], default=None)

    parser.add_argument("--soa", type=int, required=True)
    parser.add_argument("--object_activation", type=ActivationValue, required=True)
    parser.add_argument("--object_activation_duration", type=int, required=True)

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
        ),
        use_prepruned=args.sensorimotor_use_prepruned,
        filter_events=args.filter_events,
    )

    logger.info("Done!")
