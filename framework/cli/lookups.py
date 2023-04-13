"""
===========================
Lookups for command line interface.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

from framework.cognitive_model.ldm.preferences.preferences import Preferences as LDMPreferences
from framework.cognitive_model.ldm.corpus.corpus import CorpusMetadata
from framework.cognitive_model.ldm.model.base import LinguisticDistributionalModel
from framework.cognitive_model.ldm.model.count import LogCoOccurrenceCountModel
from framework.cognitive_model.ldm.model.ngram import LogNgramModel, PPMINgramModel, PMINgramModel
from framework.evolution.corpora import FILTERED_CORPORA


def get_corpus_from_name(name: str) -> CorpusMetadata:

    # Special name
    if name.lower() == "bnc":
        return LDMPreferences.source_corpus_metas.bnc
    elif name.lower() == "bbc" or name.lower() == "subtitles":
        return LDMPreferences.source_corpus_metas.bbc
    elif name.lower == "ukwac":
        return LDMPreferences.source_corpus_metas.ukwac

    for _vocab, corpus in FILTERED_CORPORA.items():
        if corpus.name.lower() == name.lower():
            return corpus

    raise LookupError(name)


def get_model_from_params(corpus: CorpusMetadata, model_name: str, radius: int) -> LinguisticDistributionalModel:
    model_name = model_name.lower()
    model_name = model_name.replace("-", "_")
    if model_name in ["log_coocc", "log_cooccurrence", "log_co-occurrence", "log_co_occurrence"]:
        return LogCoOccurrenceCountModel(corpus, radius)
    elif model_name == "log_ngram":
        return LogNgramModel(corpus, radius)
    elif model_name == "ppmi_ngram":
        return PPMINgramModel(corpus, radius)
    elif model_name == "pmi_ngram":
        return PMINgramModel(corpus, radius)
    elif model_name == "ppmi_ngram":
        return PPMINgramModel(corpus, radius)
    # TODO: FINISH THIS!!!
    else:
        raise NotImplementedError(model_name)
