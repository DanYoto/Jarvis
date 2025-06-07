import json
import queue
import sounddevice as sd
import vosk
import sys
import threading
import time
import numpy as np
from typing import Optional, Dict, List, Callable
import re  
import os


class VoskRealtimeSTT:
    """real-time speech-to-text using Vosk"""
    
    def __init__(self, 
                 model_path: str,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 device: Optional[int] = None,
                 callback: Optional[Callable] = None):
        """
        initialize real-time speech-to-text system
        
        Args:
            model_path: Vosk model path
            sample_rate: 16000
            channels: channel count, 1 for mono
            device: None for default input device, or specify device index
            callback: function to handle recognition results
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.callback = callback
        
        # processing parameters
        self.blocksize = int(sample_rate * 0.1)  # 100ms blocks
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
        # initialize Vosk
        vosk.SetLogLevel(-1)  # shut down Vosk logging
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, sample_rate)
        self.rec.SetWords(True)
        
        # Vosk 内置的部分结果超时设置
        # 设置部分结果的超时时间，单位是秒
        self.rec.SetPartialWords(True)
        self.rec.SetMaxAlternatives(0)
        
        # 用于跟踪静音
        self.last_result_time = time.time()
        self.silence_timeout = 1.5  # 1.5秒静音后认为说话结束
        
        # threading state
        self.is_recording = False
        self.processing_thread = None
        self.stream = None
        
        # Add pause functionality
        self.is_paused = False
        self.pause_lock = threading.Lock()
        
        # setup post-processing parameters
        self.enable_post_processing = True
        self.confidence_threshold = 0.6
        
        # 跟踪是否有未完成的语音
        self.has_partial_result = False
        
    def setup_audio_stream(self):
        """setup audio input stream"""
        # check device availability
        if self.device is None:
            device_info = sd.query_devices(kind='input')
        else:
            device_info = sd.query_devices(self.device, kind='input')

        print(f"Using audio device: {device_info['name']}")

        # setup audio stream
        self.stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype=np.int16,
            callback=self._audio_callback
        )
            
    def _audio_callback(self, indata, frames, time_info, status):
        """audio input callback"""
        if status:
            print(f"Audio callback status: {status}")
            
        # Check if paused
        with self.pause_lock:
            if self.is_paused:
                return
        
        # process audio data
        audio_data = indata.flatten().astype(np.int16)

        # Use put_nowait to avoid blocking in callback
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # Drop oldest audio if queue is full
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass

    def _reset_recognizer(self):
        """Safely reset the recognizer"""
        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(True)
        self.rec.SetPartialWords(True)
        self.rec.SetMaxAlternatives(0)
        self.has_partial_result = False
        self.last_result_time = time.time()

    def _process_audio_chunk(self, audio_data: np.ndarray) -> Optional[Dict]:
        """process audio chunk and perform speech recognition"""
        # Check if paused
        with self.pause_lock:
            if self.is_paused:
                return None
        
        # Convert audio data to bytes
        audio_bytes = audio_data.astype(np.int16).tobytes()
        
        try:
            # Feed audio to Vosk
            if self.rec.AcceptWaveform(audio_bytes):
                # Got final result
                result = json.loads(self.rec.Result())
                if result.get('text', '').strip():
                    self.last_result_time = time.time()
                    self.has_partial_result = False
                    return self._post_process_result(result)
            else:
                # Check partial result
                partial_result = json.loads(self.rec.PartialResult())
                partial_text = partial_result.get('partial', '').strip()
                
                if partial_text:
                    self.has_partial_result = True
                    self.last_result_time = time.time()
                    return {
                        'type': 'partial',
                        'text': partial_text,
                        'confidence': 0.5,
                        'timestamp': time.time()
                    }
                else:
                    # No partial result, check if we should finalize
                    if self.has_partial_result:
                        time_since_last = time.time() - self.last_result_time
                        if time_since_last > self.silence_timeout:
                            # Force finalization due to silence
                            self.rec.FinalResult()
                            result = json.loads(self.rec.Result())
                            self.has_partial_result = False
                            
                            if result.get('text', '').strip():
                                return self._post_process_result(result)
                            
        except Exception as e:
            print(f"Error in recognition: {e}")
            self._reset_recognizer()
            return None
        
        return None
    
    def _post_process_result(self, result: Dict) -> Dict:
        """post-process recognition result"""
        text = result.get('text', '').strip()
        
        if not text:
            return None
        
        # Get confidence (Vosk doesn't always provide confidence)
        confidence = result.get('confidence', 1.0)
        
        # Check confidence threshold
        if hasattr(self, 'confidence_threshold') and confidence < self.confidence_threshold:
            return None
        
        if self.enable_post_processing:
            text = self._enhance_text(text)
        
        # reconstruct final result
        processed_result = {
            'type': 'final',
            'text': text,
            'confidence': confidence,
            'timestamp': time.time(),
            'words': result.get('result', [])
        }
        
        return processed_result
    
    def _enhance_text(self, text: str) -> str:
        """text enhancement"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize sentences
        sentences = re.split(r'([.!?]+)', text)
        enhanced_parts = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence:
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    enhanced_parts.append(sentence)
                    
                    # Add punctuation back if it exists
                    if i + 1 < len(sentences):
                        enhanced_parts.append(sentences[i + 1])
        
        text = ''.join(enhanced_parts)
        
        # Add period if missing and text is long enough
        if text and not text[-1] in '.!?' and len(text) > 10:
            text += '.'
        
        return text
    
    def _processing_worker(self):
        """audio processing thread worker"""
        while self.is_recording:
            try:
                # get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Check if paused
                with self.pause_lock:
                    if self.is_paused:
                        self.audio_queue.task_done()
                        continue
                
                result = self._process_audio_chunk(audio_data)
                
                if result:
                    self.result_queue.put(result)
                    if self.callback:
                        self.callback(result)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Check if we need to finalize due to timeout
                if self.has_partial_result:
                    time_since_last = time.time() - self.last_result_time
                    if time_since_last > self.silence_timeout:
                        try:
                            # Force finalization
                            self.rec.FinalResult()
                            result = json.loads(self.rec.Result())
                            self.has_partial_result = False
                            
                            if result.get('text', '').strip():
                                processed_result = self._post_process_result(result)
                                if processed_result:
                                    self.result_queue.put(processed_result)
                                    if self.callback:
                                        self.callback(processed_result)
                        except Exception as e:
                            print(f"Error finalizing result: {e}")
                            self._reset_recognizer()
                continue
            except Exception as e:
                print(f"Error in processing worker: {e}")
                continue
    
    def pause_recognition(self):
        """Pause recognition without stopping the stream"""
        with self.pause_lock:
            self.is_paused = True
            # Clear the audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            # Reset recognizer state
            self._reset_recognizer()
    
    def resume_recognition(self):
        """Resume recognition"""
        with self.pause_lock:
            self.is_paused = False
            self.last_result_time = time.time()
    
    def start_recognition(self):
        """start real-time speech recognition"""
        try:
            if not hasattr(self, 'stream') or self.stream is None:
                self.setup_audio_stream()

            self.is_recording = True
            self.is_paused = False
            self.last_result_time = time.time()
            
            # Clear any old data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start processing thread if not already running
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self._processing_worker)
                self.processing_thread.daemon = True
                self.processing_thread.start()
            
            # Start audio stream
            if self.stream and not self.stream.active:
                self.stream.start()
            
        except Exception as e:
            self.stop_recognition()
            raise
    
    def stop_recognition(self):
        """stop real-time speech recognition"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Stop the stream
        if hasattr(self, 'stream') and self.stream:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive() and threading.current_thread() is not self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_results(self) -> List[Dict]:
        """get recognition results"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def set_vocabulary(self, vocabulary: List[str]):
        """create custom&limited vocabulary"""
        vocab_json = json.dumps(vocabulary)
        # dont set vocabulary
        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(True)
        self.rec.SetPartialWords(True)

    def get_word_timestamps(self, result: Dict) -> List[Dict]:
        """get word-level timestamps from recognition result"""
        words = result.get('result', [])
        return [
            {
                'word': word.get('word', ''),
                'start': word.get('start', 0),
                'end': word.get('end', 0),
                'confidence': word.get('conf', 1.0)
            }
            for word in words if word.get('word')
        ]


def result_callback(result: Dict):
    """result callback function"""
    if result['type'] == 'final':
        print(f"🎯 Final: {result['text']}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")

        words = result.get('words', [])
        if words and len(words) > 0:
            print("   Word timestamps:")
            for word in words[:5]:  # Show first 5 words
                start = word.get('start', 0)
                end = word.get('end', 0) 
                word_text = word.get('word', '')
                conf = word.get('conf', 1.0)
                print(f"     {word_text}: {start:.2f}s-{end:.2f}s (conf:{conf:.2f})")
    
    elif result['type'] == 'partial':
        print(f"⚡ Partial: {result['text']}")


def main():
    """Test the Vosk STT system"""
    model_paths = {
        'english': '/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/speech2text/models/vosk-model-en-us-0.42-gigaspeech',
        'english-small': '/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/speech2text/models/vosk-model-en-us-0.22',
    }
    
    # Try the larger model first, fallback to smaller
    model_path = model_paths.get('english')
    
    # Create STT instance
    stt = VoskRealtimeSTT(
        model_path=model_path,
        sample_rate=16000,
        callback=result_callback
    )
    
    try:
        print("🎤 Starting Vosk real-time speech recognition...")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        print("-" * 50)
        
        stt.start_recognition()
        
        while True:
            time.sleep(0.1)
            
            # Optionally process results from queue
            # results = stt.get_results()
            # for result in results:
            #     print(f"[Queue] {result}")
    
    except KeyboardInterrupt:
        print("\n🛑 User interrupted, stopping...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        stt.stop_recognition()
        print("✅ System stopped successfully")


if __name__ == "__main__":
    main()