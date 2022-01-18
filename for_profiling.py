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

from pathlib import Path
from sys import argv
from typing import Optional

from numpy import nan
from pandas import DataFrame

from os import path
from framework.cognitive_model.preferences.config import Config as ModelConfig
ModelConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))
from framework.cognitive_model.ldm.preferences.config import Config as LDMConfig
LDMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))
from framework.cognitive_model.sensorimotor_norms.config.config import Config as SMConfig
SMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "wayland_config_override.yaml"))

from framework.cli.job import CategoryVerificationJobSpec, LinguisticPropagationJobSpec, SensorimotorPropagationJobSpec
from framework.cognitive_model.attenuation_statistic import AttenuationStatistic
from framework.cognitive_model.basic_types import ActivationValue, Component
from framework.cognitive_model.combined_cognitive_model import InteractiveCombinedCognitiveModel
from framework.cognitive_model.components import FULL_ACTIVATION
from framework.cognitive_model.events import ItemEnteredBufferEvent
from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
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
from framework.data.substitution import substitutions_for
from framework.evaluation.column_names import CLOCK, CATEGORY_ACTIVATION_LINGUISTIC_f, \
    CATEGORY_ACTIVATION_SENSORIMOTOR_f, OBJECT_ACTIVATION_LINGUISTIC_f, OBJECT_ACTIVATION_SENSORIMOTOR_f

# Shared
_SN = SensorimotorNorms(use_breng_translation=True)  # Always use the BrEng translation in the interactive model


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
                for part in modified_word_tokenize(category_label_sensorimotor)
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


def main(job_spec: CategoryVerificationJobSpec):

    # Set up output directories
    response_dir: Path = Path(Preferences.output_dir, "Category verification profiling", job_spec.output_location_relative())
    if not response_dir.is_dir():
        logger.info(f"{response_dir} directory does not exist; making it.")
        response_dir.mkdir(parents=True)
    activation_tracking_dir = Path(response_dir, "activation traces")
    buffer_entries_dir = Path(response_dir, "buffer entries")
    activation_tracking_dir.mkdir(exist_ok=True)
    buffer_entries_dir.mkdir(exist_ok=True)

    # Set up model
    model = InteractiveCombinedCognitiveModel(
        sensorimotor_component=job_spec.sensorimotor_spec.to_component(SensorimotorComponent),
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

    object_activation_increment: ActivationValue = job_spec.object_activation / job_spec.incremental_activation_duration

    def _activate_sensorimotor_item(label, activation):
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


    profiling_data = [
        ("aircraft", "banana")
    ]

    for category_label, object_label in profiling_data:

        logger.info(f"Running model on {category_label} -> {object_label}")

        activation_tracking_path = Path(activation_tracking_dir, f"{category_label}-{object_label} activation.csv")
        buffer_entries_path = Path(buffer_entries_dir, f"{category_label}-{object_label} buffer.csv")

        object_label_linguistic, object_label_sensorimotor = substitutions_for(object_label)
        category_label_linguistic, category_label_sensorimotor = substitutions_for(category_label)

        category_label_linguistic_multiword_parts = modified_word_tokenize(category_label_linguistic)
        object_label_linguistic_multiword_parts = modified_word_tokenize(object_label_linguistic)

        object_prevalence: float
        try:
            object_prevalence = scale_prevalence_01(
                prevalence_from_fraction_known(_SN.fraction_known(object_label_sensorimotor)))
        except WordNotInNormsError:
            # In case the word isn't in the norms, make that known, but fall back to full prevalence
            object_prevalence = 1.0
            logger.warning(f"Could not look up prevalence as {object_label} is not in the sensorimotor norms.")
            logger.warning(f"\tUsing a default of {object_prevalence} instead.")

        model.reset()

        # Activate the initial category label in the linguistic component only
        try:
            model.linguistic_component.propagator.activate_items_with_labels(
                labels=category_label_linguistic_multiword_parts,
                activation=FULL_ACTIVATION / len(category_label_linguistic_multiword_parts))
        except ItemNotFoundError:
            logger.error(f"Missing linguistic item for category: {category_label} ({category_label_linguistic})")
            # Missing item, can make no sensible prediction
            continue

        # Start the clock
        activation_tracking_data = []
        buffer_entries = []
        back_out: bool = False  # Yes, this is ugly and fragile, but since Python doesn't have named loops I can't find a more readable way to do it.
        while model.clock <= job_spec.run_for_ticks:

            # Apply incremental activation during the immediate post-SOA period in both components
            # We won't do this on the first tick, but we do want to check and make sure that it won't fail in advance
            if object_label_sensorimotor not in model.sensorimotor_component.available_labels:
                logger.error(f"Missing sensorimotor item for object: {object_label} ({object_label_sensorimotor})")
                # Missing item, can make no sensible prediction
                back_out = True
                break
            for part in object_label_linguistic_multiword_parts:
                if part not in model.linguistic_component.available_labels:
                    logger.error(f"Missing linguistic items for object: {object_label} ({object_label_linguistic})")
                    # Missing item, can make no sensible prediction
                    back_out = True
                    break
            if back_out:
                break
            # Do the actual incremental activation
            if job_spec.soa_ticks <= model.clock < job_spec.soa_ticks + job_spec.incremental_activation_duration:
                # Activate sensorimotor item directly
                _activate_sensorimotor_item(
                    label=object_label_sensorimotor,
                    activation=object_activation_increment)
                # Activate linguistic items separately
                model.linguistic_component.propagator.activate_items_with_labels(
                    labels=object_label_linguistic_multiword_parts,
                    # Attenuate linguistic activation by object label prevalence
                    activation=object_activation_increment * object_prevalence / len(object_label_linguistic_multiword_parts))

            # Advance the model
            tick_events = model.tick()

            # Record buffer entries
            buffer_events = [e for e in tick_events if isinstance(e, ItemEnteredBufferEvent)]
            buffer_entries.extend([
                {
                    "Clock": e.time,
                    "Item ID": e.item.idx,
                    "Item label": (
                        (model.sensorimotor_component if e.item.component == Component.sensorimotor
                         else model.linguistic_component)
                        .propagator.idx2label[e.item.idx]),
                    "Activation": e.activation,
                    "Component": e.item.component,
                }
                for e in buffer_events
            ])

            logger.info(f"Clock = {model.clock}")

            # Record the relevant activations
            # There are a variable number of columns, depending on whether the items contain multiple words or not.
            # Therefore we record it in a list[dict] and build it into a DataFrame later for saving.
            activation_tracking_data.append(_get_activation_data(model,
                                            category_label_linguistic_multiword_parts, category_label_sensorimotor,
                                            object_label_linguistic_multiword_parts, object_label_sensorimotor))

        if back_out:
            continue

        with activation_tracking_path.open("w") as file:
            DataFrame(activation_tracking_data).to_csv(file, index=False)

        with buffer_entries_path.open("w") as file:
            DataFrame(buffer_entries).to_csv(file, index=False)

    Path(response_dir, " MODEL RUN COMPLETE").touch()


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(argv))

    rft = 500

    main(
        job_spec=CategoryVerificationJobSpec(
            linguistic_spec=LinguisticPropagationJobSpec(
                accessible_set_threshold=0.35,
                accessible_set_capacity=3_000,
                use_activation_cap=True,
                corpus_name="bbc",
                firing_threshold=0.35,
                length_factor=10,
                model_name="ppmi_ngram",
                node_decay_factor=0.95,
                model_radius=5,
                edge_decay_sd=15.0,
                n_words=40_000,
                pruning=None,
                pruning_type=None,
                bailout=20_000,
                run_for_ticks=rft,
            ),
            sensorimotor_spec=SensorimotorPropagationJobSpec(
                accessible_set_threshold=0.4,
                accessible_set_capacity=3000,
                use_activation_cap=True,
                distance_type=DistanceType.Minkowski3,
                length_factor=162,
                node_decay_median=5.0,
                node_decay_sigma=0.9,
                attenuation_statistic=AttenuationStatistic.Prevalence,
                max_radius=1.5,
                use_breng_translation=True,
                bailout=20_000,
                run_for_ticks=rft,
            ),
            buffer_threshold=0.4,
            buffer_capacity_linguistic_items=12,
            buffer_capacity_sensorimotor_items=9,
            cross_component_attenuation=1.0,
            lc_to_smc_delay=21,
            smc_to_lc_delay=56,
            lc_to_smc_threshold=0.4,
            smc_to_lc_threshold=0.4,
            bailout=20_000,
            run_for_ticks=rft,
            soa_ticks=241,
            object_activation=0.3,
            incremental_activation_duration=121,
        ),
    )

    logger.info("Done!")
