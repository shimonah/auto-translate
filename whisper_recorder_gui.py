#!/usr/bin/env python3

import os
import sys
import time
import threading
import queue
import datetime
import whisper
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QLineEdit, 
                            QPushButton, QTextEdit, QCheckBox, QSpinBox, 
                            QFileDialog, QGroupBox, QTabWidget, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QTextCursor

# Import functions from the original script
from whisper_recorder import get_timestamp, process_audio

class AudioRecorderThread(QThread):
    """Thread for recording audio"""
    update_status = pyqtSignal(str)
    recording_finished = pyqtSignal(str)
    
    def __init__(self, output_file, duration=None, device="default"):
        super().__init__()
        self.output_file = output_file
        self.duration = duration
        self.device = device
        self.process = None
        self.is_recording = False
        
    def run(self):
        self.is_recording = True
        self.update_status.emit(f"Recording audio to {self.output_file}...")
        
        # Build the ffmpeg command
        cmd = ["ffmpeg", "-f", "pulse", "-i", self.device, "-ac", "1", "-ar", "16000"]
        
        if self.duration:
            cmd.extend(["-t", str(self.duration)])
        
        cmd.extend(["-y", self.output_file])  # -y to overwrite existing files
        
        self.update_status.emit(f"Running command: {' '.join(cmd)}")
        
        # Start the recording process
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.process.wait()
        
        self.is_recording = False
        self.recording_finished.emit(self.output_file)
    
    def stop(self):
        if self.process and self.is_recording:
            self.process.terminate()
            self.process.wait()
            self.is_recording = False

class TranscriptionThread(QThread):
    """Thread for processing audio with Whisper"""
    update_status = pyqtSignal(str)
    update_original_text = pyqtSignal(str)
    update_translation_text = pyqtSignal(str)
    processing_finished = pyqtSignal()
    
    def __init__(self, audio_file, model, master_original_file, master_translation_file, 
                 language=None, chunk_num=None, keep_audio=False):
        super().__init__()
        self.audio_file = audio_file
        self.model = model
        self.master_original_file = master_original_file
        self.master_translation_file = master_translation_file
        self.language = language
        self.chunk_num = chunk_num
        self.keep_audio = keep_audio
        
    def run(self):
        self.update_status.emit(f"Processing {self.audio_file}...")
        
        try:
            # Transcribe in original language
            transcribe_options = {}
            if self.language and self.language != "auto":
                transcribe_options["language"] = self.language
            
            result = self.model.transcribe(self.audio_file, **transcribe_options)
            original_text = result["text"].strip()
            
            self.update_status.emit(f"Original text: {original_text[:100]}...")
            
            # Append to master original file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n\n[Chunk {self.chunk_num} - {timestamp}]\n" if self.chunk_num else f"\n\n[{timestamp}]\n"
            
            with open(self.master_original_file, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(original_text)
            
            self.update_status.emit(f"Appended original text to {self.master_original_file}")
            self.update_original_text.emit(header + original_text)
            
            # Translate to English
            result = self.model.transcribe(self.audio_file, task="translate")
            translation_text = result["text"].strip()
            
            self.update_status.emit(f"English text: {translation_text[:100]}...")
            
            # Append to master translation file
            with open(self.master_translation_file, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(translation_text)
            
            self.update_status.emit(f"Appended translation to {self.master_translation_file}")
            self.update_translation_text.emit(header + translation_text)
            
            # Remove the temporary audio file if not keeping
            if not self.keep_audio:
                try:
                    os.remove(self.audio_file)
                    self.update_status.emit(f"Removed temporary audio file: {self.audio_file}")
                except Exception as e:
                    self.update_status.emit(f"Warning: Could not remove audio file {self.audio_file}: {e}")
            
        except Exception as e:
            self.update_status.emit(f"Error processing audio: {e}")
        
        self.processing_finished.emit()

class WhisperRecorderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Recorder")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.model = None
        self.audio_queue = queue.Queue()
        self.chunk_num = 0
        self.is_recording = False
        self.continuous_mode = False
        self.recorder_thread = None
        self.transcription_thread = None
        
        # Create the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create the settings panel
        self.create_settings_panel()
        
        # Create the transcription display
        self.create_transcription_display()
        
        # Create the status bar
        self.statusBar().showMessage("Ready")
        
        # Initialize output directory
        self.output_dir = os.path.expanduser("~")
        
    def create_settings_panel(self):
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("medium")
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        
        # Language selection
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "english", "spanish", "french", "german", 
                                      "italian", "portuguese", "russian", "japanese", 
                                      "chinese", "korean", "arabic"])
        language_layout.addWidget(self.language_combo)
        settings_layout.addLayout(language_layout)
        
        # Audio device
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Audio Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["default", "default.monitor"])
        device_layout.addWidget(self.device_combo)
        settings_layout.addLayout(device_layout)
        
        # Session name
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("Session Name:"))
        self.session_name = QLineEdit(f"session_{get_timestamp()}")
        session_layout.addWidget(self.session_name)
        settings_layout.addLayout(session_layout)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_button = QPushButton("Choose...")
        self.output_dir_button.clicked.connect(self.choose_output_dir)
        output_dir_layout.addWidget(self.output_dir_button)
        settings_layout.addLayout(output_dir_layout)
        
        # Recording options
        options_layout = QHBoxLayout()
        
        # Continuous mode
        self.continuous_checkbox = QCheckBox("Continuous Mode")
        self.continuous_checkbox.stateChanged.connect(self.toggle_continuous_mode)
        options_layout.addWidget(self.continuous_checkbox)
        
        # Chunk size
        options_layout.addWidget(QLabel("Chunk Size (sec):"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(5, 300)
        self.chunk_size_spin.setValue(30)
        self.chunk_size_spin.setEnabled(False)
        options_layout.addWidget(self.chunk_size_spin)
        
        # Keep audio files
        self.keep_audio_checkbox = QCheckBox("Keep Audio Files")
        options_layout.addWidget(self.keep_audio_checkbox)
        
        settings_layout.addLayout(options_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        # Load model button
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        buttons_layout.addWidget(self.load_model_button)
        
        # Start/Stop button
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.toggle_recording)
        self.start_button.setEnabled(False)  # Disabled until model is loaded
        buttons_layout.addWidget(self.start_button)
        
        settings_layout.addLayout(buttons_layout)
        
        settings_group.setLayout(settings_layout)
        self.main_layout.addWidget(settings_group)
        
    def create_transcription_display(self):
        # Create a tab widget for original and translated text
        self.tab_widget = QTabWidget()
        
        # Original text tab
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.tab_widget.addTab(self.original_text, "Original")
        
        # Translation text tab
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.tab_widget.addTab(self.translation_text, "English Translation")
        
        # Status log tab
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.tab_widget.addTab(self.status_log, "Log")
        
        self.main_layout.addWidget(self.tab_widget, 1)  # Give it a stretch factor of 1
        
    def choose_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", 
                                                   self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.log_status(f"Output directory set to: {self.output_dir}")
    
    def toggle_continuous_mode(self, state):
        self.continuous_mode = bool(state)
        self.chunk_size_spin.setEnabled(self.continuous_mode)
    
    def load_model(self):
        self.log_status(f"Loading Whisper model: {self.model_combo.currentText()}")
        try:
            self.model = whisper.load_model(self.model_combo.currentText())
            self.log_status("Model loaded successfully")
            self.start_button.setEnabled(True)
            self.load_model_button.setEnabled(False)
            self.model_combo.setEnabled(False)
        except Exception as e:
            self.log_status(f"Error loading model: {e}")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate session name if empty
        if not self.session_name.text():
            self.session_name.setText(f"session_{get_timestamp()}")
        
        # Create master files for combined transcriptions
        self.master_original_file = os.path.join(self.output_dir, f"{self.session_name.text()}_master_original.txt")
        self.master_translation_file = os.path.join(self.output_dir, f"{self.session_name.text()}_master_english.txt")
        
        # Add headers to master files
        with open(self.master_original_file, "w", encoding="utf-8") as f:
            f.write(f"# Master Original Transcription - {self.session_name.text()}\n")
            f.write(f"# Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.language_combo.currentText() != "auto":
                f.write(f"# Language: {self.language_combo.currentText()}\n")
        
        with open(self.master_translation_file, "w", encoding="utf-8") as f:
            f.write(f"# Master English Translation - {self.session_name.text()}\n")
            f.write(f"# Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.log_status(f"Created master files:\n- {self.master_original_file}\n- {self.master_translation_file}")
        
        # Clear text displays
        self.original_text.clear()
        self.translation_text.clear()
        
        # Update UI
        self.is_recording = True
        self.start_button.setText("Stop Recording")
        self.chunk_num = 1
        
        # Disable settings during recording
        self.disable_settings(True)
        
        # Start recording
        if self.continuous_mode:
            self.log_status(f"Starting continuous recording with {self.chunk_size_spin.value()} second chunks")
            self.start_continuous_recording()
        else:
            self.log_status("Starting single recording session")
            self.start_single_recording()
    
    def start_continuous_recording(self):
        if not self.is_recording:
            return
            
        timestamp = get_timestamp()
        audio_file = os.path.join(self.output_dir, f"temp_audio_chunk_{timestamp}.wav")
        
        # Start recording thread
        self.recorder_thread = AudioRecorderThread(
            audio_file, 
            self.chunk_size_spin.value(), 
            self.device_combo.currentText()
        )
        self.recorder_thread.update_status.connect(self.log_status)
        self.recorder_thread.recording_finished.connect(self.process_recorded_audio)
        self.recorder_thread.start()
        
        self.log_status(f"Recording chunk {self.chunk_num}...")
    
    def start_single_recording(self):
        timestamp = get_timestamp()
        audio_file = os.path.join(self.output_dir, f"temp_recording_{timestamp}.wav")
        
        # Start recording thread
        self.recorder_thread = AudioRecorderThread(
            audio_file,
            None,  # No duration limit
            self.device_combo.currentText()
        )
        self.recorder_thread.update_status.connect(self.log_status)
        self.recorder_thread.recording_finished.connect(self.process_recorded_audio)
        self.recorder_thread.start()
    
    def process_recorded_audio(self, audio_file):
        # Start transcription thread
        self.transcription_thread = TranscriptionThread(
            audio_file,
            self.model,
            self.master_original_file,
            self.master_translation_file,
            self.language_combo.currentText() if self.language_combo.currentText() != "auto" else None,
            self.chunk_num if self.continuous_mode else None,
            self.keep_audio_checkbox.isChecked()
        )
        self.transcription_thread.update_status.connect(self.log_status)
        self.transcription_thread.update_original_text.connect(self.append_original_text)
        self.transcription_thread.update_translation_text.connect(self.append_translation_text)
        self.transcription_thread.processing_finished.connect(self.on_transcription_finished)
        self.transcription_thread.start()
    
    def on_transcription_finished(self):
        if self.continuous_mode and self.is_recording:
            # Increment chunk number and start next recording
            self.chunk_num += 1
            self.start_continuous_recording()
        elif not self.continuous_mode:
            # Single recording mode - we're done
            self.stop_recording()
    
    def stop_recording(self):
        self.is_recording = False
        self.start_button.setText("Start Recording")
        
        # Stop recording thread if active
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop()
            self.recorder_thread.wait()
        
        # Add end timestamp to master files
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.master_original_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n# Ended: {end_time}\n")
        with open(self.master_translation_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n# Ended: {end_time}\n")
        
        self.log_status(f"Recording session ended. Master files updated.")
        
        # Re-enable settings
        self.disable_settings(False)
    
    def disable_settings(self, disabled):
        self.language_combo.setEnabled(not disabled)
        self.device_combo.setEnabled(not disabled)
        self.session_name.setEnabled(not disabled)
        self.output_dir_button.setEnabled(not disabled)
        self.continuous_checkbox.setEnabled(not disabled)
        self.chunk_size_spin.setEnabled(not disabled and self.continuous_mode)
        self.keep_audio_checkbox.setEnabled(not disabled)
    
    def log_status(self, message):
        self.status_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
        self.statusBar().showMessage(message.split('\n')[0])
        
        # Scroll to the bottom
        cursor = self.status_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.status_log.setTextCursor(cursor)
    
    def append_original_text(self, text):
        self.original_text.append(text)
        cursor = self.original_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.original_text.setTextCursor(cursor)
    
    def append_translation_text(self, text):
        self.translation_text.append(text)
        cursor = self.translation_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.translation_text.setTextCursor(cursor)
    
    def closeEvent(self, event):
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Wait for threads to finish
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.wait()
        
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.wait()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhisperRecorderGUI()
    window.show()
    sys.exit(app.exec_()) 