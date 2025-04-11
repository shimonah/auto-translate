Uses
https://github.com/openai/whisper

Install Whisper
- pip install -U openai-whisper
- pip install faster-whisper
- sudo apt update && sudo apt install ffmpeg
- restart/reload

Run

python3 whisper_recorder.py --model=small --continuous --device default.monitor --chunk-size=30 --language russian --session-name "my_meeting"

List of models

tiny
base
small
medium
large

Obervations
- small is good with chunks 15 seconds 

Command Russian
python3 whisper_recorder.py --model=small --continuous --device default.monitor --chunk-size=15 --language ru --session-name "my_meeting" --device-type cpu --compute-type int8

Command Portuguese

python3 whisper_recorder.py --model=small --continuous --device default.monitor --chunk-size=15 --language pt --session-name "my_meeting" --device-type cpu --compute-type int8
