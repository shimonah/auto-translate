#!/usr/bin/env python3

import os
import time
import argparse
import subprocess
import whisper
import datetime
import signal
import sys
import wave

def get_timestamp():
    """Generate a timestamp for filenames"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def check_audio_file(audio_file):
    """Check if the audio file is valid and contains data"""
    try:
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            duration = frames / wf.getframerate()
            print(f"Audio file info: {frames} frames, {duration:.2f} seconds, {wf.getnchannels()} channels")
            return frames > 0 and duration > 0.5  # At least half a second of audio
    except Exception as e:
        print(f"Error checking audio file: {e}")
        return False

def record_audio(output_file, duration=None, device="default"):
    """Record audio using ffmpeg"""
    print(f"Recording audio to {output_file}...")
    
    # Build the ffmpeg command - using default.monitor to capture system audio
    cmd = ["ffmpeg", "-f", "pulse", "-i", device, "-ac", "1", "-ar", "16000"]
    
    if duration:
        cmd.extend(["-t", str(duration)])
    
    cmd.extend(["-y", output_file])  # -y to overwrite existing files
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Start the recording process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def process_audio(audio_file, model, master_original_file, master_translation_file, language=None, chunk_num=None):
    """Process audio with Whisper for transcription and translation"""
    print(f"Processing {audio_file}...")
    
    # Check if the audio file is valid
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} does not exist")
        return False
    
    if not check_audio_file(audio_file):
        print(f"Warning: Audio file {audio_file} may be empty or invalid")
    
    try:
        # Transcribe in original language (auto-detect or specified language)
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
        
        result = model.transcribe(audio_file, **transcribe_options)
        original_text = result["text"].strip()
        
        print(f"Original text: {original_text[:100]}...")
        
        # Append to master original file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(master_original_file, "a", encoding="utf-8") as f:
            if chunk_num:
                f.write(f"\n\n[Chunk {chunk_num} - {timestamp}]\n")
            else:
                f.write(f"\n\n[{timestamp}]\n")
            f.write(original_text)
        print(f"Appended original text to {master_original_file}")
        
        # Translate to English
        result = model.transcribe(audio_file, task="translate")
        translation_text = result["text"].strip()
        
        print(f"English text: {translation_text[:100]}...")
        
        # Append to master translation file
        with open(master_translation_file, "a", encoding="utf-8") as f:
            if chunk_num:
                f.write(f"\n\n[Chunk {chunk_num} - {timestamp}]\n")
            else:
                f.write(f"\n\n[{timestamp}]\n")
            f.write(translation_text)
        print(f"Appended translation to {master_translation_file}")
        
        # Remove the temporary audio file
        try:
            os.remove(audio_file)
            print(f"Removed temporary audio file: {audio_file}")
        except Exception as e:
            print(f"Warning: Could not remove audio file {audio_file}: {e}")
        
        return True
    except Exception as e:
        print(f"Error processing audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Record audio, transcribe in original language, and translate to English")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--duration", type=int, help="Recording duration in seconds (if not specified, records until interrupted)")
    parser.add_argument("--continuous", action="store_true", help="Continuously record and process audio in chunks")
    parser.add_argument("--chunk-size", type=int, default=30, help="Duration of each chunk in seconds for continuous mode")
    parser.add_argument("--language", type=str, help="Specify the language of the audio (e.g., 'portuguese')")
    parser.add_argument("--device", type=str, default="default", help="PulseAudio device to record from (default for mic, default.monitor for system audio)")
    parser.add_argument("--session-name", type=str, help="Name for the recording session (used in master file names)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep temporary audio files (default is to delete them)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate session name if not provided
    session_name = args.session_name if args.session_name else f"session_{get_timestamp()}"
    
    # Create master files for combined transcriptions
    master_original_file = os.path.join(args.output_dir, f"{session_name}_master_original.txt")
    master_translation_file = os.path.join(args.output_dir, f"{session_name}_master_english.txt")
    
    # Add headers to master files
    with open(master_original_file, "w", encoding="utf-8") as f:
        f.write(f"# Master Original Transcription - {session_name}\n")
        f.write(f"# Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.language:
            f.write(f"# Language: {args.language}\n")
    
    with open(master_translation_file, "w", encoding="utf-8") as f:
        f.write(f"# Master English Translation - {session_name}\n")
        f.write(f"# Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Created master files:\n- {master_original_file}\n- {master_translation_file}")
    
    # Load Whisper model
    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)
    
    # Handle continuous recording mode
    if args.continuous:
        print(f"Starting continuous recording mode with {args.chunk_size} second chunks.")
        print("Press Ctrl+C to stop recording.")
        
        try:
            chunk_num = 1
            while True:
                timestamp = get_timestamp()
                audio_file = os.path.join(args.output_dir, f"temp_audio_chunk_{timestamp}.wav")
                
                # Record audio chunk
                process = record_audio(audio_file, args.chunk_size, args.device)
                print(f"Waiting for chunk {chunk_num} recording to complete...")
                process.wait()
                
                # Check for any errors in ffmpeg output
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"Error recording audio chunk {chunk_num}:")
                    print(stderr.decode())
                    continue
                
                # Process the recorded chunk
                success = process_audio(
                    audio_file, model, master_original_file, master_translation_file, 
                    args.language, chunk_num
                )
                
                if success:
                    print(f"Successfully processed chunk {chunk_num}")
                else:
                    print(f"Failed to process chunk {chunk_num}")
                
                # Add a small delay between chunks
                time.sleep(1)
                chunk_num += 1
                
        except KeyboardInterrupt:
            print("\nStopping continuous recording.")
            
            # Add end timestamp to master files
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(master_original_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n# Ended: {end_time}\n")
            with open(master_translation_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n# Ended: {end_time}\n")
            
            print(f"Recording session ended. Master files updated.")
    else:
        # Single recording mode
        timestamp = get_timestamp()
        audio_file = os.path.join(args.output_dir, f"temp_recording_{timestamp}.wav")
        
        # Start recording
        process = record_audio(audio_file, args.duration, args.device)
        
        if args.duration:
            # Wait for the recording to complete
            print(f"Recording for {args.duration} seconds...")
            process.wait()
            # Process the audio
            process_audio(
                audio_file, model, master_original_file, master_translation_file,
                args.language
            )
        else:
            # For manual stopping with Ctrl+C
            print("Recording... Press Ctrl+C to stop and process the audio.")
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping recording...")
                process.terminate()
                process.wait()
                # Process the audio
                process_audio(
                    audio_file, model, master_original_file, master_translation_file,
                    args.language
                )
        
        # Add end timestamp to master files
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(master_original_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n# Ended: {end_time}\n")
        with open(master_translation_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n# Ended: {end_time}\n")
        
        print(f"Recording session ended. Master files updated.")

if __name__ == "__main__":
    main() 