#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Compute some useful values.
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
from sys import argv
from typing import List

from numpy import inf

from framework.cli.lookups import get_corpus_from_name
from framework.cognitive_model.basic_types import Length, Node
from framework.cognitive_model.graph import Edge
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.model.ngram import PPMINgramModel
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.linguistic_propagator import LinguisticPropagator
from framework.cognitive_model.sensorimotor_propagator import SensorimotorPropagator
from framework.cognitive_model.utils.logging import logger
from framework.data.category_verification_data import CategoryVerificationItemData, apply_substitution_if_available


def pairwise_lengths(linguistic_length_factor: int, linguistic_words: int, sensorimotor_length_factor: int):
    sensorimotor_propagator = SensorimotorPropagator(
        distance_type=DistanceType.Minkowski3,
        length_factor=sensorimotor_length_factor,
        max_sphere_radius=1.5,
        node_decay_lognormal_median=1,  # junk
        node_decay_lognormal_sigma=1,  # junk
        use_prepruned=True,
        use_breng_translation=True,
    )

    linguistic_propagator = LinguisticPropagator(
        length_factor=linguistic_length_factor,
        n_words=linguistic_words,
        distributional_model=PPMINgramModel(get_corpus_from_name("bbc"), 5,
                                            FreqDist.load(get_corpus_from_name("bbc").freq_dist_path)),
        distance_type=None,
        node_decay_factor=1,  # junk
        edge_decay_sd=1,  # junk
        edge_pruning_type=None,
        edge_pruning=None,
    )

    # Stimuli are the same for both datasets so it doesn't matter which we use here
    cv_item_data = CategoryVerificationItemData()

    min_sensorimotor_length, max_sensorimotor_length = inf, -inf
    min_linguistic_length, max_linguistic_length = inf, -inf
    for category_label, object_label in cv_item_data.category_object_pairs():

        category_label_linguistic = apply_substitution_if_available(category_label, cv_item_data.substitutions_linguistic)
        object_label_linguistic = apply_substitution_if_available(object_label, cv_item_data.substitutions_linguistic)
        category_label_sensorimotor = apply_substitution_if_available(category_label, cv_item_data.substitutions_sensorimotor)
        object_label_sensorimotor = apply_substitution_if_available(object_label, cv_item_data.substitutions_sensorimotor)

        # Linguistic

        try:
            category_nodes = [
                Node(linguistic_propagator.label2idx[c])
                for c in category_label_linguistic.split(" ")
            ]
            object_nodes = [
                Node(linguistic_propagator.label2idx[o])
                for o in object_label_linguistic.split(" ")
            ]
        except KeyError as e:
            logger.warning(f"Missing linguistic {e.args}")
            continue
        try:
            linguistic_lengths: List[Length] = [
                linguistic_propagator.graph.edge_lengths[Edge((cn, on))]
                for cn in category_nodes
                for on in object_nodes
            ]
        except KeyError:
            # Nodes not connected
            logger.info(f"{category_label} and {object_label} not connected in linguistic graph")
            continue

        # Sensorimotor

        try:
            category_node = Node(sensorimotor_propagator.label2idx[category_label_sensorimotor])
            object_node = Node(sensorimotor_propagator.label2idx[object_label_sensorimotor])
        except KeyError as e:
            logger.warning(f"Missing sensorimotor {e.args}")
            continue
        try:
            sensorimotor_length: Length = sensorimotor_propagator.graph.edge_lengths[Edge((category_node, object_node))]
        except KeyError:
            # Nodes not connected
            logger.info(f"{category_label} and {object_label} not reachable in sensorimotor space")
            continue

        logger.info(f"{category_label} -> {object_label}.\t"
                    f"ling length: {min(linguistic_lengths)}-{max(linguistic_lengths)}; "
                    f"sm length: {sensorimotor_length}")

        min_sensorimotor_length = min(min_sensorimotor_length, sensorimotor_length)
        max_sensorimotor_length = max(max_sensorimotor_length, sensorimotor_length)
        min_linguistic_length = min(min_linguistic_length, *linguistic_lengths)
        max_linguistic_length = max(max_sensorimotor_length, *linguistic_lengths)

    logger.info(f"")
    logger.info(f"SM range {min_sensorimotor_length}-{max_sensorimotor_length}")
    logger.info(f"LING range {min_linguistic_length}-{max_linguistic_length}")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(argv))

    parser = ArgumentParser()

    parser.add_argument("--linguistic_length_factor", required=True, type=int)
    parser.add_argument("--linguistic_words", required=True, type=int)
    parser.add_argument("--sensorimotor_length_factor", required=True, type=int)

    args = parser.parse_args()

    pairwise_lengths(
        linguistic_length_factor=args.linguistic_length_factor,
        linguistic_words=args.linguistic_words,
        sensorimotor_length_factor=args.sensorimotor_length_factor,
    )

    logger.info("Done!")
