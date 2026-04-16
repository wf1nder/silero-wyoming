from collections.abc import Iterable

import regex as re

HARD_LIMIT = 350
MERGE_BUFFER_LIMIT = 20

SENTENCE_BOUNDARY_RE = re.compile(
    r"""
    (?<!\b\p{L}{1,3})
    (?<!\p{Ll}\.\p{Ll})
    ([.!?…])
    (?=\s+\p{Lu}|\s*$)
    """,
    re.VERBOSE | re.UNICODE,
)

LIST_ITEM_RE = re.compile(r"^\s*(?:(\d+)\.|([*-]))\s*(.*)", re.MULTILINE)


def post_clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s*\((.*?)\)", r", \1, ", sentence)

    def list_replacer(match):
        num, bullet, text = match.groups()
        if num:
            return f"{num}, {text}"
        return text

    sentence = LIST_ITEM_RE.sub(list_replacer, sentence)
    sentence = sentence.replace("\n", " ").replace(";", ".")
    sentence = re.sub(r"\b([\p{IsCyrillic}]{1,3})\.\s+(?=\p{Lu})", r"\1, ", sentence)
    sentence = re.sub(r"^[.,\s]+", "", sentence)
    sentence = re.sub(r'[\*«»"]', "", sentence)
    sentence = re.sub(r"\s*—\s*", ", ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\s*([,.]\s*){2,}", r"\1 ", sentence).strip()
    return sentence


class SentenceBoundaryDetector:
    def __init__(self) -> None:
        self.buffer = ""
        self.held_sentence = ""

    def _maybe_yield(self, text: str) -> Iterable[str]:
        cleaned = post_clean_sentence(text)
        if not cleaned:
            return

        if not self.held_sentence:
            self.held_sentence = cleaned
        else:
            joiner = self.held_sentence
            if joiner.endswith("."):
                joiner = joiner[:-1] + ","
            self.held_sentence = f"{joiner} {cleaned}"

        if len(self.held_sentence) >= MERGE_BUFFER_LIMIT:
            yield self.held_sentence
            self.held_sentence = ""

    def add_chunk(self, chunk: str) -> Iterable[str]:
        self.buffer += chunk

        while True:
            match = SENTENCE_BOUNDARY_RE.search(self.buffer)
            if not match:
                if len(self.buffer) > HARD_LIMIT:
                    split_pos = self.buffer.rfind(" ", 0, HARD_LIMIT)
                    if split_pos == -1:
                        split_pos = HARD_LIMIT

                    sentence = self.buffer[:split_pos]
                    yield from self._maybe_yield(sentence)
                    self.buffer = self.buffer[split_pos:]
                    continue

                break

            sep_char = match.group(1)
            sep_end_pos = match.end(1)
            sep_start_pos = match.start(1)

            if (
                sep_char == "."
                and sep_end_pos == len(self.buffer)
                and sep_start_pos > 0
                and self.buffer[sep_start_pos - 1].isdigit()
            ):
                break

            sentence_end_pos = match.end(1)
            sentence = self.buffer[:sentence_end_pos]
            yield from self._maybe_yield(sentence)
            self.buffer = self.buffer[sentence_end_pos:]

    def finish(self) -> str:
        if self.buffer:
            cleaned = post_clean_sentence(self.buffer)
            if cleaned:
                if not self.held_sentence:
                    self.held_sentence = cleaned
                else:
                    joiner = self.held_sentence
                    if joiner.endswith("."):
                        joiner = joiner[:-1] + ","
                    self.held_sentence = f"{joiner} {cleaned}"
            self.buffer = ""

        final_text = self.held_sentence
        self.held_sentence = ""
        return final_text
