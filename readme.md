### Installation
- pip install -U openai-whisper
- pip install faster-whisper
- sudo apt update && sudo apt install ffmpeg
- restart

### Script flow

- record audio stream from default.monitor
- save it to file (length can be specified by seconds)
- model transcribes audio and translates
- two files are created:
  - one with transcription
  - one with translation to English
- repeat

### List of models

- tiny
- base
- small
- medium
- large
- turbo (only for transcription)

### Obervations

FFMPEG takes 3-5 second to capture audio, hence little chunks does not work. For example saving audio chunk 2 seconds length takes 6-7 seconds for full processing.
Chunks sizes starting from 20-30 seconds are better for tranlations to keep context sane.

In next iteration better to read stream from audio directly and somehow feed it to the model to minimize latency.

### Usage

- run command
- read the output file via `tail -f filename.txt`

### Commands

Command Portuguese Translate Only

```python3 whisper_recorder.py --model=small --continuous --device default.monitor --translate-only --chunk-size=20 --language pt --session-name "my_meeting" --device-type cpu --compute-type int8```

Command Portuguese Transcribe Only

```python3 whisper_recorder.py --model=small --continuous --device default.monitor --transcribe-only --chunk-size=20 --language pt --session-name "my_meeting" --device-type cpu --compute-type int8```


