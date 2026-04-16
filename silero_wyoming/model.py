import asyncio
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import eng_to_ipa as ipa
import numpy as np
import torch
import yaml
from num2words import num2words

LOG = logging.getLogger(__name__)

MODEL_INDEX_URL = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_CHANNELS = 1

RU_V5_MODELS = {"v5_4_ru", "v5_3_ru", "v5_2_ru", "v5_ru"}
LANGUAGE_MAP = {
    "ru": "ru-RU",
    "aze": "az-AZ",
    "hye": "hy-AM",
    "bak": "ba-RU",
    "bel": "be-BY",
    "kat": "ka-GE",
    "kbd": "kbd-RU",
    "kaz": "kk-KZ",
    "xal": "xal-RU",
    "kir": "ky-KG",
    "mdf": "mdf-RU",
    "tgk": "tg-TJ",
    "tat": "tt-RU",
    "udm": "udm-RU",
    "uzb": "uz-UZ",
    "ukr": "uk-UA",
    "kjh": "kjh-RU",
    "chv": "cv-RU",
    "erz": "myv-RU",
    "sah": "sah-RU",
}


class EnglishToRussianNormalizer:
    SIMPLE_MAP = {
        "a": "э",
        "b": "б",
        "c": "к",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "г",
        "h": "х",
        "i": "и",
        "j": "дж",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "x": "кс",
        "y": "и",
        "z": "з",
    }
    EXCEPTIONS = {
        "google": "гугл",
        "apple": "эпл",
        "microsoft": "майкрософт",
        "samsung": "самсунг",
        "whatsapp": "вотсап",
        "telegram": "телеграм",
        "youtube": "ютуб",
        "instagram": "инстаграм",
        "facebook": "фэйсбук",
        "twitter": "твиттер",
        "iphone": "айфон",
        "tesla": "тесла",
        "spacex": "спэйс икс",
        "python": "пайтон",
        "api": "эйпиай",
        "wifi": "вайфай",
        "zigbee": "зигби",
        "mqtt": "эмкутити",
        "service": "сёрвис",
        "video": "видео",
        "https": "аштитипиэс",
        "http": "аштитипи",
    }
    IPA_MAP = {
        "tʃ": "ч",
        "dʒ": "дж",
        "eɪ": "эй",
        "aɪ": "ай",
        "ɔɪ": "ой",
        "aʊ": "ау",
        "oʊ": "оу",
        "ɪə": "иэ",
        "eə": "еэ",
        "ʊə": "уэ",
        "ər": "эр",
        "ɚ": "эр",
        "ˈ": "",
        "ˌ": "",
        "ː": "",
        "p": "п",
        "b": "б",
        "t": "т",
        "d": "д",
        "k": "к",
        "g": "г",
        "m": "м",
        "n": "н",
        "f": "ф",
        "v": "в",
        "s": "с",
        "z": "з",
        "h": "х",
        "l": "л",
        "r": "р",
        "w": "в",
        "j": "й",
        "ʃ": "ш",
        "ʒ": "ж",
        "ŋ": "нг",
        "θ": "с",
        "ð": "з",
        "i": "и",
        "ɪ": "и",
        "ɛ": "э",
        "æ": "э",
        "ɑ": "а",
        "ɔ": "о",
        "u": "у",
        "ʊ": "у",
        "ʌ": "а",
        "ə": "э",
    }

    def __init__(self) -> None:
        self._max_ipa_key_len = max(len(key) for key in self.IPA_MAP)

    def _convert_ipa(self, ipa_text: str) -> str:
        result = ""
        pos = 0
        while pos < len(ipa_text):
            found = False
            for length in range(self._max_ipa_key_len, 0, -1):
                chunk = ipa_text[pos : pos + length]
                if chunk in self.IPA_MAP:
                    result += self.IPA_MAP[chunk]
                    pos += length
                    found = True
                    break
            if not found:
                pos += 1
        return result

    def _transliterate(self, match: re.Match[str]) -> str:
        word = match.group(0)
        normalized = word.lower().replace("’", "'")
        if normalized in self.EXCEPTIONS:
            return self.EXCEPTIONS[normalized]

        try:
            ipa_text = ipa.convert(normalized)
            ipa_text = re.sub(r"[/]", "", ipa_text).strip()
            if "*" in ipa_text:
                raise ValueError("ipa failed")
            result = self._convert_ipa(ipa_text)
            result = re.sub(r"йй", "й", result)
            result = re.sub(r"([чшщж])ь", r"\1", result)
            return result
        except Exception:
            return "".join(self.SIMPLE_MAP.get(ch, ch) for ch in normalized)

    def normalize(self, text: str) -> str:
        return re.sub(r"\b[a-zA-Z]+(?:['’][a-zA-Z]+)*\b", self._transliterate, text)


@dataclass
class ModelMetadata:
    model_id: str
    package_url: str
    sample_rates: list[int]
    example_text: str


class SileroTts:
    _emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\u200D"
        "\uFE0F"
        "]+",
        flags=re.UNICODE,
    )
    _emoticon_pattern = re.compile(
        r"(?:(?<=\s)|^)"
        r"(?:[:;=8xX][\-oO'^]?[)(\\/|DPp]+|[)(\\/|DPp][\-oO'^]?[:;=8xX])"
        r"(?=\s|$)"
    )
    _translation_table = str.maketrans("—–−\xa0", "--- ", "=#$“”„«»<>*\"‘’‚‹›'/")
    _cleanup_pattern = re.compile(r"[^а-яА-ЯёЁa-zA-Z0-9+?!.,:;() -]+")
    _strict_cleanup_pattern = re.compile(r"[^а-яА-ЯёЁa-zA-Z0-9+?!.,; -]+")

    def __init__(
        self,
        state_dir: str,
        language: str,
        model_id: str,
        sample_rate: int,
        model_index_url: str = MODEL_INDEX_URL,
        cpu_threads: int = 4,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.state_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = self._resolve_model(language=language, model_id=model_id, model_index_url=model_index_url)
        self.sample_rate = self._choose_sample_rate(sample_rate)
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        self.language = language
        self.model_id = model_id
        self._lock = asyncio.Lock()
        self._eng_normalizer = EnglishToRussianNormalizer()
        self._cpu_threads = cpu_threads

        torch.set_num_threads(cpu_threads)

        self.model_path = self.models_dir / f"{self.model_id}.pt"
        self._ensure_model()

        LOG.info("Loading Silero model %s on cpu", self.model_id)
        self.model = torch.package.PackageImporter(str(self.model_path)).load_pickle("tts_models", "model")
        self.speakers = list(self.model.speakers)

    def _resolve_model(self, language: str, model_id: str, model_index_url: str) -> ModelMetadata:
        with urllib.request.urlopen(model_index_url, timeout=30) as response:
            data = yaml.safe_load(response.read().decode())

        try:
            latest = data["tts_models"][language][model_id]["latest"]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Silero model: language={language} model_id={model_id}") from exc

        package_url = latest.get("package")
        if not package_url:
            raise RuntimeError(f"Silero model {model_id} does not expose a .pt package URL")

        sample_rates = latest.get("sample_rate", [DEFAULT_SAMPLE_RATE])
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates]

        return ModelMetadata(
            model_id=model_id,
            package_url=package_url,
            sample_rates=sample_rates,
            example_text=latest.get("example", ""),
        )

    def _choose_sample_rate(self, requested: int) -> int:
        if requested in self.metadata.sample_rates:
            return requested
        if DEFAULT_SAMPLE_RATE in self.metadata.sample_rates:
            return DEFAULT_SAMPLE_RATE
        return max(self.metadata.sample_rates)

    def _ensure_model(self) -> None:
        if self.model_path.is_file():
            return
        LOG.info("Downloading %s to %s", self.metadata.package_url, self.model_path)
        torch.hub.download_url_to_file(self.metadata.package_url, str(self.model_path))

    def speaker_languages(self, speaker: str) -> list[str]:
        if self.model_id in RU_V5_MODELS or "_" not in speaker:
            return ["ru-RU"]
        prefix = speaker.split("_", 1)[0]
        return [LANGUAGE_MAP.get(prefix, "ru-RU")]

    def _normalize_text(self, text: str) -> str:
        text = self._emoji_pattern.sub("", text)
        text = self._emoticon_pattern.sub(" ", text)
        text = text.translate(self._translation_table)
        text = text.replace("…", ".")
        text = re.sub(r":(?!\d)", ",", text)
        text = re.sub(r"([a-zA-Zа-яА-ЯёЁ])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([a-zA-Zа-яА-ЯёЁ])", r"\1 \2", text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\+(\d)", r" плюс \1", text)

        def repl_percent(match: re.Match[str]) -> str:
            number = match.group(1).replace(",", ".")
            return f" {number} процентов "

        text = re.sub(r"(\d+([.,]\d+)?)\s*%", repl_percent, text).replace("%", " процентов ")
        text = self._normalize_numbers(text)
        text = self._eng_normalizer.normalize(text)
        text = self._cleanup_pattern.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _strict_sanitize_text(self, text: str) -> str:
        text = self._strict_cleanup_pattern.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip(" .,:;-")

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match: re.Match[str]) -> str:
            num_str = match.group(0).replace(",", ".")
            try:
                if "." in num_str:
                    integer_part_str, fractional_part_str = num_str.split(".", 1)
                    if not integer_part_str or not fractional_part_str:
                        return num_str.replace(".", "")
                    integer_words = num2words(int(integer_part_str), lang="ru")
                    fractional_words = num2words(int(fractional_part_str), lang="ru")
                    if len(fractional_part_str) == 1:
                        return f"{integer_words} и {fractional_words}"
                    if len(fractional_part_str) == 2:
                        return f"{integer_words} и {fractional_words} сотых"
                    if len(fractional_part_str) == 3:
                        return f"{integer_words} и {fractional_words} тысячных"
                    return f"{integer_words} точка {fractional_words}"
                return num2words(int(num_str), lang="ru")
            except Exception:
                return num_str

        return re.sub(r"\b\d+([.,]\d+)?\b", replace_number, text)

    def _synthesize_blocking(self, text: str, speaker_name: str) -> bytes:
        kwargs = {
            "text": text,
            "speaker": speaker_name,
            "sample_rate": self.sample_rate,
        }
        if self.model_id in RU_V5_MODELS:
            kwargs.update(
                {
                    "put_accent": True,
                    "put_yo": True,
                    "put_stress_homo": True,
                    "put_yo_homo": True,
                }
            )

        audio = self.model.apply_tts(**kwargs)
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def synthesize(self, text: str, speaker_name: str) -> bytes | None:
        normalized = self._normalize_text(text)
        if not normalized:
            return None

        async with self._lock:
            try:
                return await asyncio.to_thread(self._synthesize_blocking, normalized, speaker_name)
            except ValueError:
                fallback = self._strict_sanitize_text(normalized)
                if not fallback or fallback == normalized:
                    LOG.warning("Silero rejected normalized text: %r", normalized)
                    return None

                LOG.warning("Silero rejected normalized text, retrying with stricter sanitization: %r -> %r", normalized, fallback)
                return await asyncio.to_thread(self._synthesize_blocking, fallback, speaker_name)
