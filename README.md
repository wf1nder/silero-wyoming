# silero-wyoming

Minimal Wyoming TTS server for [Silero Models](https://github.com/snakers4/silero-models).

It keeps the wrapper small and focuses on the things Home Assistant actually needs:
- Wyoming TTS with streaming support
- dynamic voice list from the loaded Silero model
- configurable model id
- persistent on-disk model cache

## Status

The current implementation is CPU-only.

## Features

- downloads the selected Silero model into the state directory on first start
- reuses the cached model on subsequent starts
- exposes all available speakers from the loaded model
- supports Wyoming `synthesize-start/chunk/stop`
- supports plain `synthesize`
- applies `put_accent` / `put_yo` for Russian `v5*` models

## Build

Image:

```bash
docker build -t silero-wyoming:cpu .
```

## Run

```bash
docker run --rm --network host \
  -v /opt/silero-wyoming:/data \
  silero-wyoming:cpu \
  --uri tcp://0.0.0.0:10209 \
  --state-dir /data \
  --language ru \
  --model-id v5_4_ru \
  --default-speaker xenia
```

## Home Assistant

Add it as a regular `Wyoming Protocol` integration:

- host: your server IP
- port: `10209`

Then pick the new TTS engine in your voice assistant pipeline.

## CLI

```text
--uri
--state-dir
--language
--model-id
--default-speaker
--sample-rate
--cpu-threads
--samples-per-chunk
--model-index-url
--debug
--no-streaming
```

## Notes

- models are resolved via Silero's upstream `models.yml`
- the selected model is stored as `/data/models/<model_id>.pt`
- speaker availability depends on the selected Silero model
