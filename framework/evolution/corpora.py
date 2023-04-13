from pathlib import Path
from typing import Dict

from framework.cognitive_model.ldm.corpus.corpus import CorpusMetadata
from framework.evolution.vocab import Vocab, VOCABS

_corpus_dir: Path = Path("/mmfs1/storage/users/wingfiel/corpora/vocab-evolution/")
_freq_dist_dir: Path = Path("/mmfs1/storage/users/wingfiel/indexes/vocab-evolution/")

# vocab -> corpus_meta
FILTERED_CORPORA: Dict[Vocab, CorpusMetadata] = {
    vocab: CorpusMetadata(
        name=f"Subtitles_{vocab.list_name}",
        path=          Path(_corpus_dir,    f"subtitles_{vocab.list_name}.corpus").as_posix(),
        freq_dist_path=Path(_freq_dist_dir, f"subtitles_{vocab.list_name}.freqdist").as_posix(),
    )
    for _, d in VOCABS.items()
    for _, vocab in d.items()
}
