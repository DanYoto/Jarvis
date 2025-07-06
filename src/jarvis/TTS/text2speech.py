import requests
import soundfile as sf
import json
import numpy as np
import argparse
import re
from typing import List, Tuple, Generator
import cn2an
import threading
import queue
import time
import pyaudio

def normalize_numbers_chinese(text: str) -> str:
    """
    Normalize numbers in Chinese text to a consistent format.
    """
    def convert_number(match):
        number = match.group()
        try:
            # process decimal numbers
            if '.' in number:
                integer_part, decimal_part = number.split('.')
                integer_chinese = cn2an.an2cn(int(integer_part))
                decimal_chinese = '点' + ''.join([cn2an.an2cn(int(d)) for d in decimal_part])
                return integer_chinese + decimal_chinese
            
            # process integer numbers
            num = int(number)
            if num == 0:
                return '零'
            elif 1000 <= num <= 9999:
                # For years, use the digit-by-digit reading method
                if re.search(r'(19|20)\d{2}年', match.string[max(0, match.start()-2):match.end()+1]):
                    return ''.join([cn2an.an2cn(int(d)) for d in number])
                else:
                    return cn2an.an2cn(num)
            else:
                return cn2an.an2cn(num)
                
        except:
            # If conversion fails, return digit-by-digit reading method
            return ''.join([cn2an.an2cn(int(d)) if d.isdigit() else d for d in number])

    # Match numbers (including decimals)
    text = re.sub(r'\d+\.?\d*', convert_number, text)
    return text


def normalize_special_symbols(text: str) -> str:
    symbol_map = {
        "%": "百分之",
        "‰": "千分之",
        "℃": "摄氏度",
        "°": "度",
        "km": "公里",
        "m": "米",
        "kg": "千克",
        "g": "克",
        "ml": "毫升",
        "l": "升",
        "cm": "厘米",
        "mm": "毫米",
        "km/h": "公里每小时",
        "m/s": "米每秒",
        "km²": "平方公里",
        "m²": "平方米",
        "ha": "公顷",
        "t": "吨",
        '&': '和',
        '@': '艾特',
        '#': '井号',
        '$': '美元',
        '¥': '人民币',
        '€': '欧元',
        '£': '英镑',
        '+': '加',
        '-': '减',
        '×': '乘以',
        '÷': '除以',
        '=': '等于',
        '<': '小于',
        '>': '大于',
        '≤': '小于等于',
        '≥': '大于等于',
        '℉': '华氏度'
    }

    for symbol, replacement in symbol_map.items():
        text = text.replace(symbol, replacement)
    return text


def preprocess_text(text: str) -> str:
    """
    Process the input text to normalize numbers and special symbols.
    """
    # Normalize numbers
    text = normalize_numbers_chinese(text)
    
    # Normalize special symbols
    text = normalize_special_symbols(text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def split_text_intelligently(text: str, max_length: int = 256) -> List[str]:
    """
    Split text into chunks intelligently, grouping every two sentences together.
    """

    text = preprocess_text(text)

    sentences = re.split(r'([。！？.!?]+)', text)
    
    # complete sentences is one sentence with punctuation
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
    
    # combine two sentences into one chunk if they fit within max_length
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
    改进的交叉淡入淡出，使用更自然的过渡曲线
    """
    if overlap_samples <= 0 or overlap_samples > len(audio1) or overlap_samples > len(audio2):
        return np.concatenate([audio1, audio2])
    
    fade_curve = np.linspace(0, 1, overlap_samples)
    # sigmoid-like
    fade_in = 0.5 * (1 + np.tanh(6 * (fade_curve - 0.5)))
    fade_out = 1 - fade_in
    
    audio1_fade = audio1.copy()
    audio2_fade = audio2.copy()
    
    audio1_end_rms = np.sqrt(np.mean(audio1[-overlap_samples:]**2))
    audio2_start_rms = np.sqrt(np.mean(audio2[:overlap_samples]**2))
    
    if audio1_end_rms > 0 and audio2_start_rms > 0:
        volume_ratio = audio1_end_rms / audio2_start_rms
        if volume_ratio > 2.0 or volume_ratio < 0.5:
            adjustment = np.sqrt(volume_ratio) if volume_ratio > 1 else 1/np.sqrt(1/volume_ratio)
            audio2_fade[:overlap_samples] *= min(adjustment, 1.5)
    
    audio1_fade[-overlap_samples:] *= fade_out
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
    

def synthesis_speech(server_url: str = "http://localhost:8000", 
         reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。", 
         target_text: str = None, 
         reference_audio: str = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/TTS/reference_audio/prompt_audio.wav", 
         model_name: str = "spark_tts",
         chunk_size: int = 200,
         overlap_duration: float = 0.1):

    server_url = server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    
    
    print(f"📝 Target text length: {len(target_text)} characters")
    
    try:
        print("🚀 Starting streaming audio generation and playback...")
        
        player = StreamingAudioPlayer(sample_rate=16000)
        player.start_playback()
        
        try:
            chunk_count = 0
            for audio_chunk in generate_streaming_audio(
                server_url=server_url,
                model_name=model_name,
                reference_audio_path=reference_audio,
                reference_text=reference_text,
                target_text=target_text,
                chunk_size=chunk_size,
                overlap_duration=overlap_duration,
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
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # reference_audio = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/TTS/reference_audio/prompt_audio.wav"
    # reference_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。"

    reference_audio = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/TTS/reference_audio/yanglan_zh.wav"
    reference_text = "语音合成技术其实早已经悄悄走进了我们的生活，从智能语音助手到有声读物再到个性化语音复刻。这项技术正在改变我们获取信息，与世界互动的方式，而且它的进步速度远超我们的想象。"

    # target_text = "微风拂过湖面，掀起层层涟漪，映出斑斓的夕阳余晖。远处的青山在薄雾中若隐若现，似乎在诉说着古老的传说。岸边的垂柳低垂枝条，伴随着鸟鸣轻轻摇曳，犹如一曲悠扬的古琴。几只白鹭时而起飞，振翅划过天际，又悠然落回水面，留下几声清脆的“咕咕”。这一刻，天地静谧，心境澄明，仿佛所有的烦恼都被这温柔的景色轻轻带走。"
    target_text = "春天的阳光透过窗棂洒在桌案上，微风轻拂过院中的桃花树，花瓣纷纷扬扬地飘落在青石板上。远处传来鸟儿清脆的啁啾声，仿佛在诉说着季节更替的喜悦。老人坐在藤椅上，手中捧着一杯热茶，茶香袅袅升起，与花香交融在一起，营造出一片宁静祥和的氛围。时光在这一刻仿佛放慢了脚步，让人不禁沉醉在这份简单而美好的生活中。"
    synthesis_speech(
        server_url="http://localhost:8000",
        reference_audio=reference_audio,
        reference_text=reference_text,
        target_text=target_text,
        model_name="spark_tts",
        chunk_size=40,
        overlap_duration=0.1
    )