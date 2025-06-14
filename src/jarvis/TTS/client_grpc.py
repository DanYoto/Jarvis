import requests
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import time


class SparkTTSCurlClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.model_name = "spark_tts"
    
    def health_check(self):
        """check server health"""
        try:
            response = requests.get(f"{self.server_url}/v2/health/ready", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_model_info(self):
        """get model info"""
        try:
            response = requests.get(f"{self.server_url}/v2/models/{self.model_name}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get model info: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def create_synthetic_audio(self, duration_sec=0.1, sample_rate=16000):
        """create synthetic audio data (long enough to avoid convolution kernel errors)"""
        num_samples = int(duration_sec * sample_rate)
        # create a simple sine wave + noise
        t = np.linspace(0, duration_sec, num_samples)
        frequency = 440 
        waveform = 0.1 * np.sin(2 * np.pi * frequency * t) + 0.02 * np.random.normal(0, 1, num_samples)
        return waveform.astype(np.float32)
    
    def load_audio_from_file(self, audio_path, target_sample_rate=16000):
        """load audio from file"""
        try:
            waveform, sample_rate = sf.read(audio_path)
            print(f"Loaded audio: {len(waveform)} samples at {sample_rate} Hz")

            # convert to mono
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)

            # resample
            if sample_rate != target_sample_rate:
                from scipy.signal import resample
                num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
                waveform = resample(waveform, num_samples)
                print(f"Resampled to: {len(waveform)} samples at {target_sample_rate} Hz")
            
            return waveform.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def prepare_audio_data(self, audio_path=None, min_length=1600):
        """prepare audio data"""
        if audio_path and Path(audio_path).exists():
            waveform = self.load_audio_from_file(audio_path)
            if waveform is not None and len(waveform) >= min_length:
                return waveform

        # If no audio file or too short, create synthetic audio
        print("Using synthetic audio data...")
        return self.create_synthetic_audio(duration_sec=max(0.1, min_length/16000))
    
    def synthesize(self, target_text, reference_audio_path=None, reference_text="", timeout=60):
        """
        speech synthesis

        Args:
            target_text (str): text to synthesize
            reference_audio_path (str): path to reference audio file, can be None
            reference_text (str): reference text, can be empty
            timeout (int): request timeout (seconds)

        Returns:
            tuple: (numpy.ndarray, bool) audio data and success flag
        """
        print(f"\n🎤 Starting synthesis...")
        print(f"Target text: '{target_text}'")
        print(f"Reference text: '{reference_text}'")
        print(f"Reference audio: {reference_audio_path}")

        # Prepare audio data
        waveform = self.prepare_audio_data(reference_audio_path)
        audio_length = len(waveform)
        print(f"Audio data prepared: {audio_length} samples")

        # Construct request data (fully compatible with Triton format)
        request_data = {
            "inputs": [
                {
                    "name": "reference_wav",
                    "shape": [1, audio_length],
                    "datatype": "FP32",
                    "data": [waveform.tolist()]
                },
                {
                    "name": "reference_wav_len",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[audio_length]]
                },
                {
                    "name": "reference_text",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [reference_text]
                },
                {
                    "name": "target_text",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [target_text]
                }
            ],
            "outputs": [
                {
                    "name": "waveform"
                }
            ]
        }
        
        try:
            print("📡 Sending request to server...")
            start_time = time.time()

            # Send POST request (equivalent to curl -X POST)
            response = requests.post(
                f"{self.server_url}/v2/models/{self.model_name}/infer",
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=timeout
            )
            
            end_time = time.time()
            print(f"⏱️  Request completed in {end_time - start_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()

                # Extract audio data
                if "outputs" in result and len(result["outputs"]) > 0:
                    output_data = result["outputs"][0]
                    audio_data = np.array(output_data["data"], dtype=np.float32)

                    # Handle different output formats
                    if len(audio_data.shape) > 1:
                        audio_flat = audio_data.flatten()
                    else:
                        audio_flat = audio_data
                    
                    print(f"✅ Synthesis successful!")
                    print(f"   Generated {len(audio_flat)} audio samples")
                    print(f"   Output shape: {output_data.get('shape', 'unknown')}")
                    print(f"   Duration: {len(audio_flat) / 16000:.2f} seconds")
                    
                    return audio_flat, True
                else:
                    print("❌ No audio data in response")
                    print(f"Response structure: {list(result.keys())}")
                    return None, False
                    
            else:
                print(f"❌ Synthesis failed: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Error response: {response.text}")
                return None, False
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout after {timeout} seconds")
            return None, False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            print("💡 Make sure the server is running and accessible")
            return None, False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None, False
    
    def save_audio(self, audio_data, output_path, sample_rate=16000):
        """save audio file"""
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Normalize audio data (prevent clipping)
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val * 0.95
                print(f"📊 Normalized audio (max was {max_val:.3f})")

            # Save as WAV file
            sf.write(output_path, audio_data, sample_rate, "PCM_16")
            print(f"💾 Audio saved to: {output_path}")

            # Show file info
            file_size = Path(output_path).stat().st_size
            duration = len(audio_data) / sample_rate
            print(f"   📁 File size: {file_size / 1024:.1f} KB")
            print(f"   ⏱️  Duration: {duration:.2f} seconds")
            print(f"   🔊 Sample rate: {sample_rate} Hz")
            
            return True
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return False
    
    def batch_synthesize(self, text_list, output_dir="outputs", reference_audio=None, reference_text=""):
        """batch synthesize multiple texts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        for i, text in enumerate(text_list):
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{len(text_list)}: {text[:50]}...")
            
            audio, success = self.synthesize(
                target_text=text,
                reference_audio_path=reference_audio,
                reference_text=reference_text
            )
            
            if success and audio is not None:
                output_file = output_dir / f"output_{i+1:03d}.wav"
                if self.save_audio(audio, output_file):
                    results.append((text, str(output_file), True))
                else:
                    results.append((text, None, False))
            else:
                results.append((text, None, False))
        
        return results


def main():
    print("🚀 Spark-TTS Curl-based Client")
    print("=" * 60)

    # Create client
    client = SparkTTSCurlClient()

    # Health check
    print("🔍 Checking server health...")
    if not client.health_check():
        print("❌ Server is not ready!")
        print("💡 Make sure Triton server is running:")
        print("   curl http://localhost:8000/v2/health/ready")
        return
    print("✅ Server is healthy")

    # Get model info
    print("\n📋 Getting model info...")
    model_info = client.get_model_info()
    if model_info:
        print(f"✅ Model: {model_info['name']}")
        print(f"   Versions: {model_info['versions']}")
        print(f"   Platform: {model_info.get('platform', 'unknown')}")

        # Show input/output info
        if 'inputs' in model_info:
            print("   Inputs:")
            for inp in model_info['inputs']:
                print(f"     - {inp['name']}: {inp['datatype']} {inp['shape']}")
        
        if 'outputs' in model_info:
            print("   Outputs:")
            for out in model_info['outputs']:
                print(f"     - {out['name']}: {out['datatype']} {out['shape']}")
    else:
        print("⚠️  Could not get model info, but continuing...")

    # Test cases
    test_cases = [
        {
            "name": "Simple English",
            "text": "Hello world, this is a test.",
            "output": "output_simple_english.wav"
        },
        {
            "name": "Chinese Test",
            "text": "你好世界，这是一个语音合成测试。",
            "output": "output_chinese.wav"
        },
        {
            "name": "Long English Sentence",
            "text": "This is a longer sentence to test the speech synthesis capabilities. The model should handle various sentence lengths and produce high-quality audio output.",
            "output": "output_long_english.wav"
        },
        {
            "name": "Technical Text",
            "text": "Speech synthesis using deep learning models has advanced significantly in recent years, enabling more natural and expressive voice generation.",
            "output": "output_technical.wav"
        }
    ]

    # Run tests
    print(f"\n🧪 Running {len(test_cases)} test cases...")
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        
        audio, success = client.synthesize(
            target_text=test_case['text'],
            reference_audio_path=None,  # Use synthesized audio
            reference_text=""
        )
        
        if success and audio is not None:
            if client.save_audio(audio, test_case['output']):
                success_count += 1
                print(f"✅ Test {i} completed successfully")
            else:
                print(f"❌ Test {i} failed to save audio")
        else:
            print(f"❌ Test {i} failed")

    # Summary
    print(f"\n🎯 Results Summary")
    print("=" * 40)
    print(f"Successful: {success_count}/{len(test_cases)}")
    
    if success_count > 0:
        print(f"\n🎉 Generated audio files:")
        for test_case in test_cases:
            output_path = Path(test_case['output'])
            if output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"   📄 {test_case['output']} ({size_kb:.1f} KB)")
        
        print(f"\n💡 You can play the audio files with:")
        print(f"   - macOS: afplay output_simple_english.wav")
        print(f"   - Linux: aplay output_simple_english.wav")
        print(f"   - Windows: start output_simple_english.wav")

    # If batch processing is needed
    print(f"\n📝 For batch processing, you can use:")
    print(f"   texts = ['Text 1', 'Text 2', 'Text 3']")
    print(f"   results = client.batch_synthesize(texts)")


if __name__ == "__main__":
    main()