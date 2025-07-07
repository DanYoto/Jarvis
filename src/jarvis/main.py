from speech2text.speech2text import VoskRealtimeSTT
from LLM_agents.LLMAgents import LLMClient
from TTS.text2speech import synthesis_speech
import os
import time
from typing import Dict


def main():
    # vosk model path
    vosk_model_path = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/speech2text/models/vosk-model-cn-0.22"

    # Check if model path exists
    if not os.path.exists(vosk_model_path):
        print(f"❌ Model path doesn't exist: {vosk_model_path}")
        return

    llm = LLMClient(api_key=os.getenv("llm_api_key"), deployment_name="gpt-4o")

    stt = VoskRealtimeSTT(model_path=vosk_model_path, sample_rate=16000, callback=None)

    # Define callback function to handle STT results
    def callback_pause_and_query_llm(result: Dict):
        if result["type"] == "final":
            recognized_text = result["text"]
            confidence = result.get("confidence", 1.0)
            print(f"🎯 [STT Final] Text: {recognized_text} (conf={confidence:.2f})")

            # Pause STT recognition instead of stopping
            print("⏸️ Pausing STT for LLM processing...")
            stt.pause_recognition()

            # Send recognized text to LLM
            print("🤖 Querying LLM...")
            try:
                reply = llm.chat(recognized_text)
                if reply:
                    print(f"🤖 [LLM Reply] {reply}")
                    print("Generating TTS audio...")
                    try:
                        synthesis_speech(target_text=reply)
                        print(time.time())
                        print(f"✅ TTS audio generated and played successfully")
                    except Exception as tts_error:
                        print(f"❌ TTS error: {tts_error}")
                else:
                    print("⚠️ LLM did not return any content or the call failed.")
            except Exception as e:
                print(f"❌ LLM error: {e}")

            # Resume STT after LLM processing is done
            print("▶️ LLM processing complete, resuming STT")
            stt.resume_recognition()

        elif result["type"] == "partial":
            # Optional: show partial results
            # print(f"⚡ [STT Partial] {result['text']}")
            pass

    # Set the callback function
    stt.callback = callback_pause_and_query_llm

    try:
        print("🎤 Starting Vosk real-time speech recognition...")
        print("Speak something, then pause. Press Ctrl+C to stop.")
        print("-" * 50)

        stt.start_recognition()
        start_ts = time.time()

        while True:
            time.sleep(0.1)

            # Optional: get and process results from queue
            # This is redundant if callback is set, but useful for debugging
            # results = stt.get_results()
            # for result in results:
            #     print(f"[Queue] {result}")

            # Print status every 30 seconds
            elapsed = time.time() - start_ts
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"⏱️ Running time: {elapsed:.0f}s")
                # Sleep to avoid multiple prints
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 User pressed Ctrl+C, exiting...")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Ensure STT is completely stopped on exit
        print("🧹 Cleaning up...")
        stt.stop_recognition()
        print("✅ System stopped successfully")


if __name__ == "__main__":
    main()
