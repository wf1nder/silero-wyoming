# silero-wyoming

Minimal Wyoming TTS server for [Silero Models](https://github.com/snakers4/silero-models).

This image is focused on a simple CPU-only deployment for Home Assistant and other Wyoming-compatible clients.

## Features

- Wyoming TTS server
- Streaming support (`synthesize-start/chunk/stop`)
- Plain `synthesize` support
- Dynamic speaker list from the loaded Silero model
- Dynamic model download on first start
- Persistent on-disk model cache
- Russian `v5*` model handling with `put_accent` and `put_yo`

## Quick Start

```bash
docker run --rm --network host \
  -v /opt/silero-wyoming:/data \
  wf1nder/silero-wyoming:latest \
  --uri tcp://0.0.0.0:10209 \
  --state-dir /data \
  --language ru \
  --model-id v5_4_ru \
  --default-speaker xenia
```

## Useful Options

- `--uri`
  Wyoming listen address, for example `tcp://0.0.0.0:10209`
- `--state-dir`
  Directory for cached model files inside the container
- `--language`
  Silero language key, for example `ru`
- `--model-id`
  Silero model id, for example `v5_4_ru`, `v5_3_ru`, `v5_ru`
- `--default-speaker`
  Default speaker name from the loaded model
- `--sample-rate`
  Output sample rate
- `--cpu-threads`
  Number of CPU threads for PyTorch
- `--samples-per-chunk`
  PCM chunk size for Wyoming streaming
- `--model-index-url`
  Override upstream `models.yml` location
- `--no-streaming`
  Disable Wyoming streaming support
- `--debug`
  Enable verbose logging

## Home Assistant

Add it as a regular `Wyoming Protocol` integration:

- Host: your server IP
- Port: `10209`

Then select the new TTS engine in your voice assistant pipeline.

## Notes

- Models are not baked into the image.
- The selected model is downloaded on first start and stored in `/data/models/<model_id>.pt`.
- Speaker availability depends on the selected Silero model.
- The current image is CPU-only.

## Source

GitHub repository: [wf1nder/silero-wyoming](https://github.com/wf1nder/silero-wyoming)
