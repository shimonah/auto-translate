Uses
https://github.com/openai/whisper

Install Whisper
- pip install -U openai-whisper
- sudo apt update && sudo apt install ffmpeg
- restart/reload

Run

python3 whisper_recorder.py --continuous --device default.monitor --language portuguese --session-name "my_meeting"

