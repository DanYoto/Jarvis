import os
import argparse
import torch
import pygame
import threading
import soundfile as sf
from datetime import datetime
from .SparkTTS.cli.SparkTTS import SparkTTS
import numpy as np
from io import BytesIO
import time 
import platform

class TTSRunner:
    """
    Text-to-Speech runner that encapsulates model loading and inference.
    """
    def __init__(
        self,
        model_dir: str = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/TTS/models/SparkAudioTTS",
        save_dir: str = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/TTS/output",
        device_index: int = 0,
        gender: str = 'female',
        pitch: str = 'moderate',
        speed: str = 'moderate',
    ):
        """
        Initialize the TTSRunner.

        Args:
            model_dir (str): Path to the TTS model directory.
            save_dir (str): Directory to save generated audio.
            device_index (int, optional): CUDA or MPS device index. Defaults to 0.
            gender (str, optional): Voice gender ('male' or 'female').
            pitch (str, optional): Pitch level.
            speed (str, optional): Speech speed.
        """
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.device_index = device_index
        self.gender = gender
        self.pitch = pitch
        self.speed = speed
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = self._select_device()
        # Initialize the model
        self.model = SparkTTS(self.model_dir, self.device)

    def _select_device(self) -> torch.device:
        """
        Determine the appropriate torch.device (CPU, CUDA, or MPS).

        Returns:
            torch.device: The selected device.
        """
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_index}")
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            print('end macos')
            device = torch.device(f"mps:{self.device_index}")
        else:
            device = torch.device("cpu")
        return device

    def run(
        self,
        text: str,
        prompt_speech_path: str = None,
        prompt_text: str = None,
        save_to_file: bool = False,
    ) -> str:
        """
        Perform TTS inference and save the generated audio wav file.

        Args:
            text (str): Input text to synthesize.
            prompt_speech_path (str, optional): Path to reference audio file for style prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Returns:
            str: The file path of the saved audio.
        """

        with torch.no_grad():
            wav = self.model.inference(
                text,
                prompt_speech_path,
                prompt_text=prompt_text,
                gender=self.gender,
                pitch=self.pitch,
                speed=self.speed,
            )

            sample_rate = 16000
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = os.path.join(self.save_dir, f"{timestamp}.wav")
                sf.write(save_path, wav, samplerate=sample_rate)

        return wav, sample_rate, save_path if save_to_file else None


class AudioPlayer:
    """
    Simple audio player to play generated audio.
    """
    def __init__(self, sample_rate: int = 16000):
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer = 1024)
        self.lock = threading.Event()
        self.is_initialized = True
    
    def play(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Play the given audio data.

        Args:
            audio_data (np.ndarray): Audio data to play.
        """
        if not self.is_initialized:
            raise RuntimeError("Audio player is not initialized.")
        
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        buffer = BytesIO()
        sf.write(buffer, audio, samplerate=sample_rate, format='WAV')
        buffer.seek(0)

        self.lock.clear()
        pygame.mixer.music.load(buffer)
        pygame.mixer.music.play()

        # Wait until playback ends
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    def stop(self) -> None:
        pygame.mixer.music.stop()

    def shutdown(self) -> None:
        pygame.mixer.quit()
        self.is_initialized = False



if __name__ == "__main__":
    # Initialize TTS runner
    runner = TTSRunner()
    # Initialize audio player
    player = AudioPlayer(sample_rate=16000)
    
    try:
        # Generate speech without saving to file
        print("Generating speech...")
        audio_data, sample_rate, _ = runner.run(
            text="Hello, world! This is an audio playback test.",
            prompt_speech_path=None,
            prompt_text=None,
            save_to_file=False,  # Don't save to file
        )
        
        print(f"Speech generation completed, sample rate: {sample_rate} Hz")
        print(f"Audio data shape: {audio_data.shape}")
        
        # Play the generated audio
        print("Playing audio...")
        player.play(audio_data, sample_rate)
        print("Playback completed!")
        
        # Test file saving functionality
        print("\nGenerating and saving another speech segment...")
        audio_data2, sample_rate2, save_path = runner.run(
            text="This is the second test speech that will be saved to a file.",
            prompt_speech_path=None,
            prompt_text=None,
            save_to_file=True,  # Save to file
        )
        
        if save_path:
            print(f"Audio saved to: {save_path}")
            
        # Play the second audio segment
        print("Playing second audio segment...")
        player.play(audio_data2, sample_rate2)
        print("Playback completed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown audio player
        player.shutdown()
        print("Audio player shutdown complete")