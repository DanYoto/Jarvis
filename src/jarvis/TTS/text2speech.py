import requests
import soundfile as sf
import json
import numpy as np
import argparse
import re
from typing import List, Tuple, Generator
import io
import wave
import threading
import queue
import time
import pyaudio

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../example/prompt_audio.wav",
        help="Path to a single audio file",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="spark_tts",
        choices=["f5_tts", "spark_tts"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio (optional, for saving final result)",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum number of characters per chunk",
    )
    
    parser.add_argument(
        "--overlap-duration",
        type=float,
        default=0.1,
        help="Duration in seconds to overlap between chunks",
    )
    
    parser.add_argument(
        "--from-file",
        type=str,
        default=None,
        help="Read target text from a file instead of command line",
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming playback while generating",
    )
    
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save the final concatenated audio file",
    )
    
    return parser.parse_args()

def split_text_intelligently(text: str, max_length: int = 200) -> List[str]:
    """
    Split text into chunks intelligently, grouping every two sentences together.
    """
    sentences = re.split(r'([。！？.!?]+)', text)
    
    complete_sentences = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence_with_punct = sentences[i] + sentences[i + 1]
            if sentence_with_punct.strip():
                complete_sentences.append(sentence_with_punct)
        else:
            if sentences[i].strip():
                complete_sentences.append(sentences[i])
    
    if len(complete_sentences) <= 1:
        print("No sentence endings found, splitting by commas...")
        parts = re.split(r'([，,、；;]+)', text)
        complete_sentences = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                part_with_punct = parts[i] + parts[i + 1]
                if part_with_punct.strip():
                    complete_sentences.append(part_with_punct)
            else:
                if parts[i].strip():
                    complete_sentences.append(parts[i])
    
    print(f"Found {len(complete_sentences)} sentence/clause units")
    
    chunks = []
    for i in range(0, len(complete_sentences), 2):
        if i + 1 < len(complete_sentences):
            two_sentences = complete_sentences[i] + complete_sentences[i + 1]
        else:
            two_sentences = complete_sentences[i]
        
        if len(two_sentences) <= max_length:
            chunks.append(two_sentences.strip())
        else:
            if len(complete_sentences[i]) <= max_length:
                chunks.append(complete_sentences[i].strip())
            else:
                sub_chunks = split_long_sentence(complete_sentences[i], max_length)
                chunks.extend(sub_chunks)
            
            if i + 1 < len(complete_sentences):
                if len(complete_sentences[i + 1]) <= max_length:
                    chunks.append(complete_sentences[i + 1].strip())
                else:
                    sub_chunks = split_long_sentence(complete_sentences[i + 1], max_length)
                    chunks.extend(sub_chunks)
    
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1} ({len(chunk)} chars): {chunk[:50]}...")
    
    return chunks

def split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """
    Split a long sentence by commas or other punctuation.
    """
    sub_parts = re.split(r'([，,、；;]+)', sentence)
    sub_chunks = []
    current_chunk = ""
    
    for part in sub_parts:
        if len(current_chunk) + len(part) <= max_length:
            current_chunk += part
        else:
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
            current_chunk = part
    
    if current_chunk:
        sub_chunks.append(current_chunk.strip())
    
    return sub_chunks

def prepare_request(waveform, reference_text, target_text, sample_rate=16000):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    
    samples = waveform.reshape(1, -1).astype(np.float32)

    data = {
        "inputs":[
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist()
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
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
        ]
    }

    return data

def apply_crossfade(audio1: np.ndarray, audio2: np.ndarray, overlap_samples: int) -> np.ndarray:
    """
    Apply crossfade between two audio segments.
    """
    if overlap_samples <= 0 or overlap_samples > len(audio1) or overlap_samples > len(audio2):
        return np.concatenate([audio1, audio2])
    
    fade_out = np.linspace(1, 0, overlap_samples)
    fade_in = np.linspace(0, 1, overlap_samples)
    
    audio1_fade = audio1.copy()
    audio1_fade[-overlap_samples:] *= fade_out
    
    audio2_fade = audio2.copy()
    audio2_fade[:overlap_samples] *= fade_in
    
    result = np.concatenate([
        audio1[:-overlap_samples],
        audio1_fade[-overlap_samples:] + audio2_fade[:overlap_samples],
        audio2[overlap_samples:]
    ])
    
    return result

class StreamingAudioPlayer:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.playing = False
        self.generation_finished = False
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.chunks_played = 0
        
    def start_playback(self):
        """Start the audio playback stream"""
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        self.playing = True
        
        # Launch playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
    def add_audio_chunk(self, audio_data: np.ndarray, chunk_index: int = None):
        """Add audio data to the playback queue"""
        if chunk_index is not None:
            print(f"📥 Adding audio chunk {chunk_index + 1} to playback queue, length: {len(audio_data)} samples")
        self.audio_queue.put(audio_data)
        
    def finish_generation(self):
        """Mark generation as complete"""
        self.generation_finished = True
        print("🏁 Audio generation complete, waiting for queue to empty...")
        
    def stop_playback(self):
        """Stop playback"""
        self.playing = False
        self.audio_queue.put(None)  # send stop signal
        
        if hasattr(self, 'playback_thread') and self.playback_thread:
            self.playback_thread.join(timeout=5.0)
            
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.pa.terminate()
        
    def wait_for_completion(self):
        """Wait for all audio playback to complete"""
        while self.playing:
            if self.generation_finished and self.audio_queue.empty():
                if not self.stream.is_active():
                    self.playing = False
                    break
            time.sleep(0.05)

        if hasattr(self, 'playback_thread'):
            self.playback_thread.join(timeout=2.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()

    def _playback_worker(self):
        """Playback worker thread"""
        print("🎵 Starting audio stream playback...")

        while self.playing:
            try:
                audio_data = self.audio_queue.get(timeout=2.0)

                if audio_data is None:
                    print("🛑 Stop signal received, exiting playback loop")
                    break

                self.chunks_played += 1
                print(f"▶️ Playing chunk {self.chunks_played}, length: {len(audio_data)} samples")

                chunk_size = self.chunk_size
                for i in range(0, len(audio_data), chunk_size):
                    if not self.playing:
                        break
                    chunk = audio_data[i:i + chunk_size]
                    self.stream.write(chunk.astype(np.float32).tobytes())

                print(f"✅ Chunk {self.chunks_played} playback finished")

            except queue.Empty:
                if self.generation_finished and self.audio_queue.empty():
                    print("⏹️ Generation complete and queue empty, ending playback")
                    break
                continue
            except Exception as e:
                print(f"❌ Playback error: {e}")
                break

        self.playing = False
        print(f"🎵 Playback finished, total chunks played: {self.chunks_played}")

        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

def generate_streaming_audio(
    server_url: str,
    model_name: str,
    reference_audio_path: str,
    reference_text: str,
    target_text: str,
    chunk_size: int = 200,
    overlap_duration: float = 0.1,
    sample_rate: int = 16000,
    streaming: bool = True,
    save_final: bool = False,
    output_path: str = None,
    player: StreamingAudioPlayer = None
) -> Generator[np.ndarray, None, None]:
    """
    Generate streaming audio, yielding audio segments as they are produced
    """
    # Load the reference audio
    reference_waveform, sr = sf.read(reference_audio_path)
    assert sr == sample_rate, f"Reference audio sample rate must be {sample_rate}"
    reference_samples = np.array(reference_waveform, dtype=np.float32)
    
    # Split the text
    chunks = split_text_intelligently(target_text, chunk_size)
    print(f"📄 Text split into {len(chunks)} chunks")
    
    # Prepare for final audio saving if needed
    all_audio_segments = [] if save_final else None
    overlap_samples = int(overlap_duration * sample_rate)
    previous_audio = None

    for i, chunk in enumerate(chunks):
        print(f"🔄 Generating chunk {i+1}/{len(chunks)}: {chunk[:30]}...")
        
        try:
            # Prepare request
            data = prepare_request(reference_samples, reference_text, chunk)
            
            # Send request
            url = f"{server_url}/v2/models/{model_name}/infer"
            start_time = time.time()
            
            rsp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                verify=False,
                params={"request_id": str(i)}
            )
            rsp.raise_for_status()
            
            generation_time = time.time() - start_time
            print(f"⚡ Chunk {i+1} generated in {generation_time:.2f}s")
            
            result = rsp.json()
            audio_data = result["outputs"][0]["data"]
            current_audio = np.array(audio_data, dtype=np.float32)
            
            # Apply crossfade except for the first chunk
            if previous_audio is not None and overlap_samples > 0:
                overlap_part = previous_audio[-overlap_samples:] if len(previous_audio) >= overlap_samples else previous_audio
                current_audio = apply_crossfade(overlap_part, current_audio, len(overlap_part))
                playback_audio = current_audio[len(overlap_part):]
            else:
                playback_audio = current_audio
            
            previous_audio = current_audio
            
            # Save segment if required
            if save_final:
                all_audio_segments.append(playback_audio)
            
            duration = len(playback_audio) / sample_rate
            print(f"✅ Chunk {i+1} ready, duration: {duration:.2f}s, samples: {len(playback_audio)}")
            
            # Stream to player if available
            if player:
                player.add_audio_chunk(playback_audio, i)
            
            # Yield the audio segment
            yield playback_audio
            
        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Signal end of generation to player
    if player:
        player.finish_generation()
    
    # Save final audio file if requested
    if save_final and all_audio_segments and output_path:
        print("💾 Saving final audio file...")
        final_audio = np.concatenate(all_audio_segments)
        sf.write(output_path, final_audio, sample_rate, "PCM_16")
        print(f"📁 Final audio saved to: {output_path}")
        print(f"⏱️ Total duration: {len(final_audio) / sample_rate:.2f} seconds")

def main():
    args = get_args()
    
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    
    # Read target text from file if specified
    if args.from_file:
        with open(args.from_file, 'r', encoding='utf-8') as f:
            target_text = f.read().strip()
    else:
        target_text = args.target_text
    
    print(f"📝 Target text length: {len(target_text)} characters")
    
    try:
        if args.streaming:
            # Streaming mode: generate and play back
            print("🚀 Starting streaming audio generation and playback...")
            
            player = StreamingAudioPlayer(sample_rate=16000)
            player.start_playback()
            
            try:
                chunk_count = 0
                for audio_chunk in generate_streaming_audio(
                    server_url=server_url,
                    model_name=args.model_name,
                    reference_audio_path=args.reference_audio,
                    reference_text=args.reference_text,
                    target_text=target_text,
                    chunk_size=args.chunk_size,
                    overlap_duration=args.overlap_duration,
                    streaming=True,
                    save_final=args.save_final,
                    output_path=args.output_audio if args.save_final else None,
                    player=player
                ):
                    chunk_count += 1
                    print(f"🎼 Generator yielded chunk {chunk_count}")
                
                print(f"🏁 All chunks generated, total: {chunk_count}")
                print("⏳ Waiting for playback to finish...")
                player.wait_for_completion()
                print("✅ Playback complete, exiting")
                
            except KeyboardInterrupt:
                print("\n⚠️ Playback interrupted by user")
            finally:
                print("🛑 Stopping player...")
                player.stop_playback()
                
        else:
            # Batch mode: generate full audio then save
            print("📁 Batch mode: generating full audio file...")
            all_chunks = list(generate_streaming_audio(
                server_url=server_url,
                model_name=args.model_name,
                reference_audio_path=args.reference_audio,
                reference_text=args.reference_text,
                target_text=target_text,
                chunk_size=args.chunk_size,
                overlap_duration=args.overlap_duration,
                streaming=False,
                save_final=True,
                output_path=args.output_audio
            ))
            
            if all_chunks:
                final_audio = np.concatenate(all_chunks)
                sf.write(args.output_audio, final_audio, 16000, "PCM_16")
                print(f"📁 Audio saved to: {args.output_audio}")
                print(f"⏱️ Total duration: {len(final_audio) / 16000:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


"""
python client_http.py \
            --reference-audio ../../example/prompt_audio.wav \
            --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
            --target-text "微风拂过湖面，掀起层层涟漪，映出斑斓的夕阳余晖。远处的青山在薄雾中若隐若现，似乎在诉说着古老的传说。岸边的垂柳低垂枝条，伴随着鸟鸣轻轻摇曳，犹如一曲悠扬的古琴。几只白鹭时而起飞，振翅划过天际，又悠然落回水面，留下几声清脆的“咕咕”。这一刻，天地静谧，心境澄明，仿佛所有的烦 恼都被这温柔的景色轻轻带走。" \
            --model-name spark_tts --streaming
"""