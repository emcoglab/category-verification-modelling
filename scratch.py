from collections import deque
from pathlib import Path

from framework.cognitive_model.ldm.corpus.corpus import StreamedCorpus
from framework.cognitive_model.ldm.preferences.preferences import Preferences


def main():
    corpus = StreamedCorpus(Preferences.source_corpus_metas.bbc.path)

    buffer_size = 20
    tokens = ["baby", "rock"]

    buffer = deque()
    example_count = 0
    with Path("/Users/caiwingfield/Desktop/baby_rock.txt").open("w") as out_file:
        for i, token in enumerate(corpus, start=1):
            buffer.append(token)
            if len(buffer) > buffer_size:  # Overfull
                buffer.popleft()
            if len(buffer) == buffer_size:  # Full
                centre = int(buffer_size/2)
                if buffer[centre - 1] == tokens[0] and buffer[centre] == tokens[1]:
                    out_file.write(" ".join(buffer) + "\n")
                    example_count += 1
            if i % 1_000_000 == 0:
                print(f"{i:,}:\t{example_count} examples")


if __name__ == '__main__':
    main()
