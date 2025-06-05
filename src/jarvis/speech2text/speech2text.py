#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vosk实时语音识别系统 - 为Jarvis优化
支持中英文混合识别，极低延迟，轻量级部署
"""

import json
import queue
import sounddevice as sd
import vosk
import sys
import threading
import time
import numpy as np
from typing import Optional, Dict, List, Callable
import webrtcvad
from scipy import signal
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.blocksize = int(sample_rate * 0.1)  # 100ms
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # initialize audio stream
        vosk.SetLogLevel(-1)  # shut down Vosk logging
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, sample_rate)
        self.rec.SetWords(True)
        logger.info(f"Vosk model load successfully: {model_path}")
        
        # initialize VAD
        self.vad = webrtcvad.Vad(2)  # medium aggressiveness
        self.vad_frame_duration = 30  # 30ms
        self.vad_frame_size = int(sample_rate * self.vad_frame_duration / 1000)
        
        # audio detection state
        self.is_speaking = False
        self.speech_frames = []
        self.silence_count = 0
        self.silence_threshold = 20  # stop after 20 consecutive silent frames
        
        # threading state
        self.is_recording = False
        self.processing_thread = None
        
        # setup post-processing parameters
        self.enable_post_processing = True
        self.confidence_threshold = 0.6
        
    def setup_audio_stream(self):
        """setup audio input stream"""
        try:
            # check device availability
            if self.device is None:
                device_info = sd.query_devices(kind='input')
                logger.info(f"Use default audio input system: {device_info['name']}")
            else:
                device_info = sd.query_devices(self.device, kind='input')
                logger.info(f"Use pointed device: {device_info['name']}")
            
            # setup audio stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype=np.int16,
                callback=self._audio_callback
            )
            
            logger.info(f"Audio Stream configured - samplerate: {self.sample_rate}Hz, blocksize: {self.blocksize}")
            
        except Exception as e:
            logger.error(f"Audio setup stream failed: {e}")
            raise
    
    def _audio_callback(self, indata, frames, time_info, status):
        """audio input callback"""
        if status:
            logger.warning(f"audio stream state: {status}")
        
        # process audio data
        audio_data = indata.flatten().astype(np.int16)
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
        else:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put(audio_data)
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """audio preprocessing"""
        # normalize audio volume
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        if len(audio_data) > 64:  # make sure audio data is long enough for filtering
            b, a = signal.butter(3, 80/(self.sample_rate/2), 'high')
            audio_data = signal.filtfilt(b, a, audio_data)
        
        return audio_data.astype(np.int16)
    
    def _detect_speech_vad(self, audio_frame: np.ndarray) -> bool:
        """use VAD to detect speech in audio frame"""
        try:
            if len(audio_frame) != self.vad_frame_size:
                return False
            
            # turn into bytes for VAD
            frame_bytes = audio_frame.astype(np.int16).tobytes()
            
            # VAD detection
            return self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as e:
            logger.debug(f"VAD detection error: {e}")
            return False
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> Optional[Dict]:
        """process audio chunk and perform speech recognition"""
        audio_data = self._preprocess_audio(audio_data)
        
        # VAD detection
        speech_detected = False
        for i in range(0, len(audio_data), self.vad_frame_size):
            frame = audio_data[i:i+self.vad_frame_size]
            if len(frame) == self.vad_frame_size:
                if self._detect_speech_vad(frame):
                    speech_detected = True
                    break
        
        # audio data handling
        if speech_detected:
            self.is_speaking = True
            self.silence_count = 0
            self.speech_frames.extend(audio_data)
        else:
            if self.is_speaking:
                self.silence_count += 1
                self.speech_frames.extend(audio_data)
                
                # when silence threshold reached, transcribe speech
                if self.silence_count >= self.silence_threshold:
                    result = self._transcribe_speech()
                    self.is_speaking = False
                    self.silence_count = 0
                    self.speech_frames = []
                    return result
        
        # real-time recognition
        if len(self.speech_frames) > 0:
            # send accumulated audio frames for recognition
            audio_bytes = np.array(self.speech_frames, dtype=np.int16).tobytes()
            
            if self.rec.AcceptWaveform(audio_bytes):
                result = json.loads(self.rec.Result())
                if result.get('text', '').strip():
                    # set buffer to empty after processing
                    self.speech_frames = []
                    return self._post_process_result(result)
            else:
                # obtain partial result
                partial_result = json.loads(self.rec.PartialResult())
                if partial_result.get('partial', '').strip():
                    return {
                        'type': 'partial',
                        'text': partial_result['partial'],
                        'confidence': 0.5,
                        'timestamp': time.time()
                    }
        
        return None
    
    def _transcribe_speech(self) -> Optional[Dict]:
        """transcribe accumulated speech frames"""
        if not self.speech_frames:
            return None
        
        try:
            # turn accumulated speech frames into bytes
            audio_bytes = np.array(self.speech_frames, dtype=np.int16).tobytes()
            
            # speech recognition
            if self.rec.AcceptWaveform(audio_bytes):
                result = json.loads(self.rec.Result())
            else:
                result = json.loads(self.rec.FinalResult())
            
            if result.get('text', '').strip():
                return self._post_process_result(result)
                
        except Exception as e:
            logger.error(f"transcribe failed: {e}")
        
        return None
    
    def _post_process_result(self, result: Dict) -> Dict:
        """post-process recognition result"""
        text = result.get('text', '').strip()
        
        if not text:
            return None
        
        # confirm confidence level
        confidence = result.get('confidence', 1.0)
        if confidence < self.confidence_threshold:
            logger.debug(f"confidence too low: {confidence}, text: {text}")
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
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = text.split('.')
        enhanced_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                enhanced_sentences.append(sentence)
        
        if enhanced_sentences:
            text = '. '.join(enhanced_sentences)
            if not text.endswith('.') and len(text) > 10:
                text += '.'
        
        return text
    
    def _processing_worker(self):
        """audio processing thread worker"""
        logger.info("audo processing thread started")
        
        while self.is_recording:
            try:
                # get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)

                result = self._process_audio_chunk(audio_data)
                
                if result:
                    self.result_queue.put(result)
                    if self.callback:
                        try:
                            self.callback(result)
                        except Exception as e:
                            logger.error(f"callback function failed: {e}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"audio process failed: {e}")
        
        logger.info("audio processing thread stopped")
    
    def start_recognition(self):
        """start real-time speech recognition"""
        if self.is_recording:
            logger.warning("real-time speech recognition is already running")
            return
        
        try:

            self.setup_audio_stream()

            self.is_recording = True
            self.processing_thread = threading.Thread(target=self._processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.stream.start()
            
            logger.info("real-time speech recognition started")
            
        except Exception as e:
            logger.error(f"recognition failed: {e}")
            self.stop_recognition()
            raise
    
    def stop_recognition(self):
        """stop real-time speech recognition"""
        if not self.is_recording:
            return
        
        logger.info("Stopping real-time speech recognition...")
        
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Stop recognition failed: {e}")
        
        # wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("real-time speech recognition stopped")
    
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
        """create custom vocabulary"""
        try:
            vocab_json = json.dumps(vocabulary, ensure_ascii=False)
            self.rec.SetGrammar(vocab_json)
            logger.info(f"customized volcabulary including {len(vocabulary)} vocabulary items")
        except Exception as e:
            logger.error(f"create custom vocalbulary failed: {e}")
    
    def get_word_timestamps(self, result: Dict) -> List[Dict]:
        """get word-level timestamps from recognition result"""
        words = result.get('words', [])
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
        print(f"🎯 final recognition: {result['text']}")
        print(f"   confidence: {result['confidence']:.2f}")

        words = result.get('words', [])
        if words:
            print("   word time step:")
            for word in words[:5]: 
                start = word.get('start', 0)
                end = word.get('end', 0) 
                word_text = word.get('word', '')
                conf = word.get('conf', 1.0)
                print(f"     {word_text}: {start:.2f}s-{end:.2f}s (置信度:{conf:.2f})")
    
    elif result['type'] == 'partial':
        print(f"⚡ real-time recognition: {result['text']}")


def main():
    model_paths = {
        'english': '/home/azureuser/cloudfiles/code/Users/yutong.jiang2/personal_file/Jarvis/src/jarvis/speech2text/models/vosk-model-small-en-us-0.15',
    }
    
    model_path = model_paths.get('english')
    
    import os
    if not os.path.exists(model_path):
        print(f"❌ model path doesnt exist: {model_path}")
        return
    
    # create VoskRealtimeSTT instance
    stt = VoskRealtimeSTT(
        model_path=model_path,
        sample_rate=16000,
        callback=result_callback
    )
    
    # set up custom vocabulary
    custom_vocabulary = [
        "Hi"
    ]
    stt.set_vocabulary(custom_vocabulary)
    
    try:
        print("🎤 start Vosk real-time audio recognition...")
        print("test, press Ctrl+C to stop")
        print("-" * 50)
        
        # 开始识别
        stt.start_recognition()
        
        # 主循环
        start_time = time.time()
        while True:
            time.sleep(0.1)
            
            # 定期显示运行状态
            if int(time.time() - start_time) % 30 == 0:
                runtime = time.time() - start_time
                print(f"⏱️  running time: {runtime:.0f}s")
                start_time = time.time()
    
    except KeyboardInterrupt:
        print("\n🛑 user interrupted, stopping...")
    
    except Exception as e:
        print(f"❌ running failure {e}")
    
    finally:
        stt.stop_recognition()
        print("✅ system stopped successfully")


if __name__ == "__main__":
    
    main()