import logging
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from .model import SileroTts
from .sentence_boundary import SentenceBoundaryDetector

LOG = logging.getLogger(__name__)


class SileroEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args,
        tts: SileroTts,
        default_speaker_name: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts = tts
        self.default_speaker_name = default_speaker_name
        self.is_streaming = False
        self.sbd: Optional[SentenceBoundaryDetector] = None
        self._stream_synthesize: Optional[Synthesize] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    LOG.debug("Ignoring legacy Synthesize during streaming session")
                    return True
                return await self._handle_synthesize(Synthesize.from_event(event))

            if SynthesizeStart.is_type(event.type):
                if not self.cli_args.streaming:
                    return True
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                stream_start = SynthesizeStart.from_event(event)
                self._stream_synthesize = Synthesize(text="", voice=stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming:
                    return True
                assert self.sbd is not None
                assert self._stream_synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    self._stream_synthesize.text = sentence
                    await self._handle_synthesize(self._stream_synthesize)
                return True

            if SynthesizeStop.is_type(event.type):
                if not self.is_streaming:
                    return True
                assert self.sbd is not None
                assert self._stream_synthesize is not None
                final_text = self.sbd.finish()
                if final_text:
                    self._stream_synthesize.text = final_text
                    await self._handle_synthesize(self._stream_synthesize)
                await self.write_event(SynthesizeStopped().event())
                self.is_streaming = False
                self.sbd = None
                self._stream_synthesize = None
                return True

            return True
        except Exception as exc:
            LOG.exception("Error while handling Wyoming event")
            self.is_streaming = False
            await self.write_event(Error(text=str(exc), code=exc.__class__.__name__).event())
            return False

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        if not synthesize.text:
            return True

        speaker_name = self.default_speaker_name
        if synthesize.voice and synthesize.voice.name in self.tts.speakers:
            speaker_name = synthesize.voice.name

        text = " ".join(synthesize.text.strip().splitlines())
        audio_bytes = await self.tts.synthesize(text=text, speaker_name=speaker_name)
        if not audio_bytes:
            return True

        await self.write_event(
            AudioStart(
                rate=self.tts.sample_rate,
                width=self.tts.sample_width,
                channels=self.tts.channels,
            ).event()
        )

        bytes_per_chunk = self.tts.sample_width * self.tts.channels * self.cli_args.samples_per_chunk
        for i in range(0, len(audio_bytes), bytes_per_chunk):
            await self.write_event(
                AudioChunk(
                    audio=audio_bytes[i : i + bytes_per_chunk],
                    rate=self.tts.sample_rate,
                    width=self.tts.sample_width,
                    channels=self.tts.channels,
                ).event()
            )

        await self.write_event(AudioStop().event())
        return True
