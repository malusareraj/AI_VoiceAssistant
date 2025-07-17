import os
import time
import threading
import queue
from openai import OpenAI
import numpy as np
from transformers import pipeline
from gtts import gTTS
import torch
import io
import soundfile as sf
import sounddevice as sd
# Removed pydub import since it's not used and causing issues
from langdetect import detect
import tempfile
import pygame
from pathlib import Path

# Configuration
SAMPLE_RATE = 16000
SAMBASTUDIO_API_KEY = "1dc8dd92-24c8-4760-a1eb-6195af9021a3"  # Replace with your actual key
RECORD_DURATION = 10  # Maximum recording duration in seconds

class SpeechToSpeechLLaMA:
    def __init__(self):
        print("Initializing Speech-to-Speech with LLaMA 8B...")
        
        # Initialize OpenAI client for Sambanova
        self.client = OpenAI(
            base_url="https://api.sambanova.ai/v1",
            api_key=SAMBASTUDIO_API_KEY
        )
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize speech recognition model
        print("Loading speech recognition model...")
        self.speech_recognizer = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Speech recognition model loaded successfully!")
        
        # Audio recording variables
        self.audio_queue = queue.Queue()
        self.recording = False
        
    def record_audio(self, duration=RECORD_DURATION):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds... Speak now!")
        
        try:
            # Record audio
            audio_data = sd.rec(int(duration * SAMPLE_RATE), 
                              samplerate=SAMPLE_RATE, 
                              channels=1, 
                              dtype=np.float32)
            sd.wait()  # Wait until recording is finished
            
            # Check if we actually got audio data
            if np.max(np.abs(audio_data)) < 0.001:
                print("Warning: Very quiet audio detected")
            
            return audio_data.flatten()
            
        except Exception as e:
            print(f"Error during recording: {e}")
            return None
    
    def record_until_silence(self, silence_threshold=0.01, max_silence_duration=2.0):
        """Record audio until silence is detected"""
        print("Recording... Speak now! (Will stop after silence)")
        
        chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        audio_chunks = []
        silence_counter = 0
        max_silence_chunks = int(max_silence_duration / 0.1)
        
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                
                # Calculate RMS (volume level)
                rms = np.sqrt(np.mean(indata**2))
                
                if rms < silence_threshold:
                    nonlocal silence_counter
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                audio_chunks.append(indata.copy())
                
                # Stop if we've had enough silence
                if silence_counter >= max_silence_chunks:
                    raise sd.CallbackStop()
            
            # Start recording
            with sd.InputStream(callback=audio_callback, 
                              channels=1, 
                              samplerate=SAMPLE_RATE,
                              blocksize=chunk_size):
                while True:
                    try:
                        sd.sleep(100)  # Sleep for 100ms
                    except sd.CallbackStop:
                        break
            
            if audio_chunks:
                # Combine all chunks
                audio_data = np.concatenate(audio_chunks, axis=0).flatten()
                print("Recording completed!")
                return audio_data
            else:
                print("No audio recorded")
                return None
                
        except Exception as e:
            print(f"Error during recording: {e}")
            return None
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio to text using Whisper"""
        try:
            # Add noise filtering to remove very quiet audio
            if np.max(np.abs(audio_data)) < 0.001:  # Very quiet audio
                return ""
            
            result = self.speech_recognizer(
                {"raw": audio_data, "sampling_rate": SAMPLE_RATE},
                return_timestamps=False
            )
            transcription = result["text"].strip()
            
            # Filter out repetitive dots or meaningless transcriptions
            if len(transcription) > 100 and transcription.count('.') / len(transcription) > 0.8:
                return ""
            
            return transcription
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def query_llama(self, prompt):
        """Send prompt to LLaMA 8B model hosted on Sambanova"""
        try:
            completion = self.client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Respond concisely to the user's query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error querying Sambanova API: {e}")
            return "I couldn't connect to the AI service. Please try again later."
    
    def detect_language(self, text):
        """Detect language of the text"""
        try:
            lang = detect(text)
            # Map detected language to gTTS language codes
            lang_map = {
                'en': 'en',    # English
                'hi': 'hi',    # Hindi
                'mr': 'mr',    # Marathi
                'es': 'es',    # Spanish
                'fr': 'fr',    # French
                'de': 'de',    # German
                'it': 'it',    # Italian
                'pt': 'pt',    # Portuguese
                'ru': 'ru',    # Russian
                'ja': 'ja',    # Japanese
                'ko': 'ko',    # Korean
                'zh': 'zh',    # Chinese
                'ar': 'ar',    # Arabic
            }
            return lang_map.get(lang, 'en')  # Default to English if unknown
        except:
            return 'en'
    
    def text_to_speech(self, text):
        """Automatically detect language and convert to speech"""
        try:
            # Detect language from the text
            lang = self.detect_language(text)
            print(f"Detected language: {lang}")
            
            # Create speech synthesis
            tts = gTTS(text=text, lang=lang)
            
            # Save to temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tmp_file.close()  # Close the file so gTTS can write to it
            
            tts.save(tmp_file.name)
            
            # Verify file exists before playing
            if os.path.exists(tmp_file.name):
                # Play the audio using pygame
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass  # Ignore cleanup errors
            else:
                print("Error: Audio file was not created")
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def process_speech_input(self, audio_data):
        """Process speech input through the complete pipeline"""
        if audio_data is None or len(audio_data) == 0:
            print("No audio data received")
            return
        
        # Transcribe speech to text
        print("Transcribing speech...")
        user_input = self.transcribe_audio(audio_data)
        
        if not user_input:
            print("No speech detected")
            self.text_to_speech("I didn't hear anything. Please try again.")
            return
        
        print(f"You said: {user_input}")
        
        # Get response from LLM
        print("Getting response from LLaMA...")
        response = self.query_llama(user_input)
        print(f"LLaMA response: {response}")
        
        # Convert response to speech
        print("Converting response to speech...")
        self.text_to_speech(response)
    
    def interactive_mode(self):
        """Run in interactive mode with voice commands"""
        print("\n=== Speech-to-Speech with LLaMA 8B ===")
        print("Commands:")
        print("  - Press ENTER to start recording")
        print("  - Type 'quit' to exit")
        print("  - Type 'auto' for automatic silence detection")
        print("  - Type 'manual' for manual recording (10 seconds)")
        print("\nReady! Press ENTER to start...")
        
        mode = "auto"  # Default mode
        
        while True:
            try:
                command = input().strip().lower()
                
                if command == 'quit':
                    print("Goodbye!")
                    break
                elif command == 'auto':
                    mode = "auto"
                    print("Switched to automatic silence detection mode")
                    continue
                elif command == 'manual':
                    mode = "manual"
                    print("Switched to manual recording mode (10 seconds)")
                    continue
                elif command == '' or command == 'record':
                    # Start recording based on current mode
                    if mode == "auto":
                        audio_data = self.record_until_silence()
                    else:
                        audio_data = self.record_audio()
                    
                    if audio_data is not None:
                        self.process_speech_input(audio_data)
                    
                    print("\nPress ENTER to record again, or type a command...")
                else:
                    print("Unknown command. Press ENTER to record, or type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Press ENTER to try again...")

def main():
    """Main function"""
    try:
        # Create speech-to-speech instance
        speech_system = SpeechToSpeechLLaMA()
        
        # Run interactive mode
        speech_system.interactive_mode()
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install openai sounddevice numpy torch transformers gtts langdetect pygame")

if __name__ == "__main__":
    main()