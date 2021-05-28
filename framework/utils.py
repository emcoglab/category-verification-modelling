from typing import List

from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize


def decompose_multiword(term: str) -> List[str]:
    return modified_word_tokenize(term)
