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
- small is good with chunks 15 seconds 

### Usage

- run command
- read the output file via `tail -f filename.txt`

### Commands

Command Portuguese Translate Only

```python3 whisper_recorder.py --model=small --continuous --device default.monitor --translate-only --chunk-size=15 --language pt --session-name "my_meeting" --device-type cpu --compute-type int8```

Command Portuguese Transcribe Only

```python3 whisper_recorder.py --model=small --continuous --device default.monitor --transcribe-only --chunk-size=15 --language pt --session-name "my_meeting" --device-type cpu --compute-type int8```


