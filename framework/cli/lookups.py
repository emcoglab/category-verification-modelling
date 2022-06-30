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


from ..cognitive_model.ldm.corpus.corpus import CorpusMetadata
from ..cognitive_model.ldm.corpus.indexing import FreqDist
from ..cognitive_model.ldm.model.base import LinguisticDistributionalModel
from ..cognitive_model.ldm.model.count import LogCoOccurrenceCountModel
from ..cognitive_model.ldm.model.ngram import LogNgramModel, PPMINgramModel, PMINgramModel
from ..cognitive_model.ldm.preferences.preferences import Preferences as CorpusPreferences


def get_corpus_from_name(name: str) -> CorpusMetadata:
    if name.lower() == "bnc":
        return CorpusPreferences.source_corpus_metas.bnc
    elif name.lower() == "bbc" or name.lower() == "subtitles":
        return CorpusPreferences.source_corpus_metas.bbc
    elif name.lower == "ukwac":
        return CorpusPreferences.source_corpus_metas.ukwac
    else:
        raise NotImplementedError(name)


def get_model_from_params(corpus: CorpusMetadata, freq_dist: FreqDist, model_name: str, radius: int) -> LinguisticDistributionalModel:
    model_name = model_name.lower()
    if model_name == "log_coocc" or model_name == "log_cooccurrence" or model_name == "log_co-occurrence":
        return LogCoOccurrenceCountModel(corpus, radius, freq_dist)
    elif model_name == "log_ngram":
        return LogNgramModel(corpus, radius, freq_dist)
    elif model_name == "ppmi_ngram":
        return PPMINgramModel(corpus, radius, freq_dist)
    elif model_name == "pmi_ngram":
        return PMINgramModel(corpus, radius, freq_dist)
    # TODO: FINISH THIS!!!
    else:
        raise NotImplementedError(model_name)
