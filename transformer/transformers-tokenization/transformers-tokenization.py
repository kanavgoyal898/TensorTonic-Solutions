class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.vocab_size = len(self.word_to_id)

        words = set()
        for text in texts:
            text = text.lower()
            words.update(set(text.split()))

        words = list(words)
        words = sorted(words)

        for word in words:
            self.word_to_id[word] = self.vocab_size
            self.vocab_size += 1

        for word, id in self.word_to_id.items():
            self.id_to_word[id] = word

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        text = text.lower()
        words = text.split()

        ids = []
        for word in words:
            ids.append(
                self.word_to_id.get(
                    word, 
                    self.word_to_id[self.unk_token]
                )
            )
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = []
        for id in ids:
            words.append(
                self.id_to_word.get(
                    id,
                    self.id_to_word[self.word_to_id[self.unk_token]]
                )
            )
        text = " ".join(words)
        return text