from pathlib import Path
from typing import Iterator, Set, Dict

from pandas import DataFrame, read_csv


class Vocab:
    """
    Vocabs have a base name which is derived from their source wordlist, but have many sub-lists used to model an
    evolution or development of the vocab in certain directions.
    """
    def __init__(self, base_name: str, source_path: Path, list_i: int):
        """
        :param: base_name: The name of the vocab, irrespective of which filter level it's using
        :param: source_path: The location of the main vocab file
        :param: list_i: which filter list it's using
        """
        self.list: int = list_i

        self.base_name: str = base_name
        self._source_path: Path = source_path

        # backs self.words property
        self._words = None

        # backs self.items property
        self._items = None

    @property
    def list_name(self) -> str:
        """The name of the vocab, including which list is used to truncate it"""
        return f"{self.base_name}_list{self.list}"

    @property
    def words(self) -> Set[str]:
        if self._words is None:
            self._words = set(self._iter_words_from_vocab(decompose_multiwords=True))
        return self._words

    @property
    def items(self) -> Set[str]:
        if self._items is None:
            self._items = set(self._iter_words_from_vocab(decompose_multiwords=False))
        return self._items

    def _iter_words_from_vocab(self, decompose_multiwords: bool) -> Iterator[str]:
        df = self._get_vocab(self._source_path, col_prefix=self.base_name[0:4])

        list_numbers = df["list.number"].unique()
        assert self.list in list_numbers

        # Lists are cumulative, so "list 3" includes words from list == 1 and list == 2
        for row_i, row in df[df["list.number"] <= self.list].iterrows():
            if decompose_multiwords:
                for word in row["Word"].split(" "):
                    yield word.strip()
            else:
                yield row["Word"]

    # region Hashable

    def __hash__(self):
        return hash((self.list_name, self._source_path, self.list))

    def __eq__(self, other):
        return isinstance(other, Vocab) and (hash(self) == hash(other))

    # endregion

    def __len__(self):
        return len(self.words)

    @staticmethod
    def _get_vocab(wordlist_path: Path, col_prefix: str = "") -> DataFrame:
        """Read the vocab source file."""
        df: DataFrame = read_csv(wordlist_path.as_posix(), header=0)
        if col_prefix:
            df.rename(columns={
                f"{col_prefix}.itemNum":     "itemNum",
                f"{col_prefix}.Word":        "Word",
                f"{col_prefix}.list.number": "list.number",
                f"{col_prefix}.rank":        "rank",
            }, inplace=True)
        df['Word'] = df['Word'].str.lower().str.strip()
        df.sort_values(by="rank", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)  # Do this rather than using sort_index=True in df.sort_values to maintain compatibility with old versions of pandas
        return df


_data_dir = Path("/mmfs1/storage/users/wingfiel/experimental_data/vocab-evolution/")
_vocab_paths: Dict[str, Path] = {
    "AoA": Path(_data_dir, "AoA vocab list N=34138.csv"),
    "Evolutionary": Path(_data_dir, "Evolutionary vocab list N=34138.csv"),
    "Frequency": Path(_data_dir, "Frequency vocab list N=34138.csv"),
}
_list_nums = [1, 2, 3, 4, 5, 6, 7]

# The canonical list of Vocabs to be used in the experiment
# base_name -> list_i -> Vocab
VOCABS: Dict[str, Dict[int, Vocab]] = {
    base_name: {
        i: Vocab(
            base_name=base_name,
            source_path=p,
            list_i=i,
        )
        for i in _list_nums
    }
    for base_name, p in _vocab_paths.items()
}
