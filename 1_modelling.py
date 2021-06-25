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

from numpy import nan
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
from framework.cognitive_model.sensorimotor_norms.breng_translation.dictionary.dialect_dictionary import ameng_to_breng
from framework.cognitive_model.sensorimotor_norms.exceptions import WordNotInNormsError
from framework.cognitive_model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from framework.cognitive_model.utils.exceptions import ItemNotFoundError
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.utils.maths import scale_prevalence_01, prevalence_from_fraction_known
from framework.data.category_verification_data import CategoryVerificationItemData, apply_substitution_if_available
from framework.evaluation.column_names import CLOCK, CATEGORY_ACTIVATION_LINGUISTIC_f, \
    CATEGORY_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f, OBJECT_ACTIVATION_SENSORIMOTOR_f
from framework.utils import decompose_multiword

# arg choices: filter_events
_ARG_ACCESSIBLE_SET = "accessible_set"
_ARG_BUFFER         = "buffer"

_sn = SensorimotorNorms(use_breng_translation=True)  # Always use the BrEng translation in the interactive model


def _get_best_sensorimotor_translation(sensorimotor_component: SensorimotorComponent, w: str) -> Optional[str]:
    """
    Returns the best available translation, or None of none are good enough.
    :param: sensorimotor_component
        The component to check for label availability.
    :param: w
        The term to attempt translation for.
    """
    if w in sensorimotor_component.available_labels:
        return w
    for translation in ameng_to_breng.best_translations_for(w):
        if translation in sensorimotor_component.available_labels:
            return translation
    return None


def _get_activation_data(model, category_multiword_parts, category_label_sensorimotor, object_multiword_parts,
                         object_label_sensorimotor):
    # We already know that linguistic category terms exist, as we've activated them (or failed to) before
    category_activation_linguistic_dict = {
        CATEGORY_ACTIVATION_LINGUISTIC_f.format(part)
        : model.linguistic_component.propagator.activation_of_item_with_label(part)
        for part in category_multiword_parts
    }
    # For the sensorimotor category, we need to check that it exists, and skip it if it doesn't
    try:
        category_activation_sensorimotor_dict = {
            CATEGORY_ACTIVATION_SENSORIMOTOR_f.format(category_label_sensorimotor)
            : model.sensorimotor_component.propagator.activation_of_item_with_label(category_label_sensorimotor)
        }
    except ItemNotFoundError:
        # If the item isn't found in total, we can check for individual components
        # We haven't done this before because we don't ever activate the category in the sensorimotor component
        # directly.
        try:
            category_activation_sensorimotor_dict = {
                CATEGORY_ACTIVATION_SENSORIMOTOR_f.format(part)
                : model.sensorimotor_component.propagator.activation_of_item_with_label(part)
                for part in decompose_multiword(category_label_sensorimotor)
            }
        except ItemNotFoundError:
            # If this item really doesn't exist, we can omit the activation
            category_activation_sensorimotor_dict = {
                CATEGORY_ACTIVATION_SENSORIMOTOR_f.format(category_label_sensorimotor)
                : nan
            }
    # We know that linguistic and sensorimotor objects exist
    object_activation_linguistic_dict = {
        OBJECT_ACTIVATION_LINGUISTIC_f.format(part)
        : model.linguistic_component.propagator.activation_of_item_with_label(part)
        for part in object_multiword_parts
    }
    object_activation_sensorimotor_dict = {
        OBJECT_ACTIVATION_SENSORIMOTOR_f.format(object_label_sensorimotor)
        : model.sensorimotor_component.propagator.activation_of_item_with_label(object_label_sensorimotor)
    }
    return {
        CLOCK: model.clock,
        **category_activation_linguistic_dict,
        **category_activation_sensorimotor_dict,
        **object_activation_linguistic_dict,
        **object_activation_sensorimotor_dict,
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

    object_activation_increment: ActivationValue = job_spec.object_activation / job_spec.incremental_activation_duration

    def activate_sensorimotor_item(label, activation):
        """Handles activation of sensorimotor item with translation if necessary."""
        if label in model.sensorimotor_component.available_labels:
            model.sensorimotor_component.propagator.activate_item_with_label(label, activation)
        else:
            logger.warning(f"Missing sensorimotor item: {label}")
            translation = _get_best_sensorimotor_translation(model.sensorimotor_component, label)
            if translation is not None:
                logger.warning(f" Attempting with translation: {translation}")
                model.sensorimotor_component.propagator.activate_item_with_label(translation, activation)
            else:
                logger.error(f" No translations available")

    for category_label, object_label in cv_item_data.category_object_pairs():

        activation_tracking_path = Path(response_dir, f"{category_label}-{object_label}.csv")
        activation_tracking_data = []

        category_label_linguistic = apply_substitution_if_available(category_label, cv_item_data.substitutions_linguistic)
        category_label_sensorimotor = apply_substitution_if_available(category_label, cv_item_data.substitutions_sensorimotor)
        object_label_linguistic = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
        object_label_sensorimotor = apply_substitution_if_available(object_label, cv_item_data.substitutions_sensorimotor)

        object_prevalence: float
        try:
            object_prevalence = scale_prevalence_01(
                prevalence_from_fraction_known(_sn.fraction_known(object_label_sensorimotor)))
        except WordNotInNormsError:
            # In case the word isn't in the norms, make that known, but fall back to full prevalence
            object_prevalence = 1.0
            logger.warning(f"Could not look up prevalence as {object_label} is not in the sensorimotor norms.")
            logger.warning(f"\tUsing a default of {object_prevalence} instead.")

        model.reset()

        # Activate the initial category label in the linguistic component only
        category_label_linguistic_multiword_parts = decompose_multiword(category_label_linguistic)
        object_label_linguistic_multiword_parts = decompose_multiword(object_label_linguistic)
        try:
            model.linguistic_component.propagator.activate_items_with_labels(
                labels=category_label_linguistic_multiword_parts,
                activation=FULL_ACTIVATION / len(category_label_linguistic_multiword_parts))
        except ItemNotFoundError:
            logger.error(f"Missing linguistic item for category: {category_label} ({category_label_linguistic})")
            # Missing item, can make no sensible prediction
            continue

        # Start the clock
        while True:
            if model.clock > job_spec.run_for_ticks:
                break

            # Apply incremental activation during the immediate post-SOA period in both components
            if job_spec.soa_ticks <= model.clock < job_spec.soa_ticks + job_spec.incremental_activation_duration:

                # Activate sensorimotor item directly
                try:
                    activate_sensorimotor_item(
                        label=object_label_sensorimotor,
                        activation=object_activation_increment)
                except ItemNotFoundError:
                    logger.error(f"Missing sensorimotor item for object: {object_label} ({object_label_sensorimotor})")
                    # Missing item, can make no sensible prediction
                    break
                # Activate linguistic items separately
                try:
                    model.linguistic_component.propagator.activate_items_with_labels(
                        labels=object_label_linguistic_multiword_parts,
                        # Attenuate linguistic activation by object label prevalence
                        activation=object_activation_increment * object_prevalence / len(object_label_linguistic_multiword_parts))
                except ItemNotFoundError:
                    logger.error(f"Missing linguistic items for object: {object_label} ({object_label_linguistic})")
                    # Missing item, can make no sensible prediction
                    break

            # Advance the model
            model.tick()
            logger.info(f"Clock = {model.clock}")

            # Record the relevant activations
            # There are a variable number of columns, depending on whether the items contain multiple words or not.
            # Therefore we record it in a list[dict] and build it into a DataFrame later for saving.
            activation_tracking_data.append(_get_activation_data(model,
                                            category_label_linguistic_multiword_parts, category_label_sensorimotor,
                                            object_label_linguistic_multiword_parts, object_label_sensorimotor))

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
