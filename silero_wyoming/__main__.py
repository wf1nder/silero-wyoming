import asyncio
import contextlib
import logging
import sys
from argparse import ArgumentParser
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SileroEventHandler
from .model import MODEL_INDEX_URL, SileroTts

ATTRIBUTION_NAME = "Silero"
ATTRIBUTION_URL = "https://github.com/snakers4/silero-models"
PROGRAM_NAME = "silero-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for Silero TTS"
PROGRAM_VERSION = "0.1.2"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10208", help="unix:// or tcp://")
    parser.add_argument("--state-dir", default="/data")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--model-id", default="v5_4_ru")
    parser.add_argument("--default-speaker", default="xenia")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument("--model-index-url", default=MODEL_INDEX_URL)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-streaming", action="store_false", dest="streaming")
    parser.set_defaults(streaming=True)
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    log = logging.getLogger(__name__)

    try:
        tts = SileroTts(
            state_dir=args.state_dir,
            language=args.language,
            model_id=args.model_id,
            sample_rate=args.sample_rate,
            model_index_url=args.model_index_url,
            cpu_threads=args.cpu_threads,
        )
    except Exception as exc:
        log.critical("Failed to initialize Silero TTS: %s", exc, exc_info=True)
        sys.exit(1)

    if args.default_speaker not in tts.speakers:
        fallback = tts.speakers[0]
        log.warning("Default speaker %s not found, using %s", args.default_speaker, fallback)
        args.default_speaker = fallback

    voices = [
        TtsVoice(
            name=speaker,
            description=speaker,
            attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
            installed=True,
            version=args.model_id,
            languages=tts.speaker_languages(speaker),
        )
        for speaker in tts.speakers
    ]

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESCRIPTION,
                attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
                installed=True,
                version=PROGRAM_VERSION,
                voices=voices,
                supports_synthesize_streaming=args.streaming,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    handler_factory = partial(
        SileroEventHandler,
        wyoming_info,
        args,
        tts,
        args.default_speaker,
    )
    log.info(
        "Starting %s on %s with model=%s speaker=%s",
        PROGRAM_NAME,
        args.uri,
        args.model_id,
        args.default_speaker,
    )
    await server.run(handler_factory)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
