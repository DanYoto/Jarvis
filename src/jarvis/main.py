from speech2text.speech2text import VoskRealtimeSTT
from LLM_agents.LLMAgents import LLMClient 
import os
import time
from typing import Dict


def main():
    # vosk model path
    model_path = "/Users/yutong.jiang2/Library/CloudStorage/OneDrive-IKEA/Desktop/Jarvis/src/jarvis/speech2text/models/vosk-model-en-us-0.22"

    llm = LLMClient(
        api_key="abs",
        model_name="gpt-3.5-turbo"
    )

    stt = VoskRealtimeSTT(
        model_path=model_path,
        sample_rate=16000,
        callback=None
    )

    # 4) define callback function to handle STT results
    #    This function will be called whenever STT has a final result.
    def callback_pause_and_query_llm(result: Dict):
        if result['type'] == 'final':
            recognized_text = result['text']
            confidence = result['confidence']
            print(f"🎯 [STT Final] Text: {recognized_text} (conf={confidence:.2f})")

            # stop STT recognition temporarily
            stt.stop_recognition()

            # send recognized text to LLM
            print("⏸️ STT is stopped, querying LLM...")
            reply = llm.query(recognized_text)
            if reply:
                print(f"🤖 [LLM Reply] {reply}")
            else:
                print("⚠️ LLM did not return any content or the call failed.")

            # when LLM processing is done, restart STT
            print("▶️ LLM processing complete, restarting STT")
            stt.start_recognition()

        elif result['type'] == 'partial':
            # print(f"⚡ [STT Partial] {result['text']}")
            pass

    # set the callback function to handle STT results
    stt.callback = callback_pause_and_query_llm

    try:
        stt.start_recognition()
        start_ts = time.time()
        while True:
            time.sleep(0.1)
            # results = stt.get_results()

            if int(time.time() - start_ts) % 30 == 0:
                runtime = time.time() - start_ts
                print(f"⏱️ running time: {runtime:.0f}s")
                start_ts = time.time()

    except KeyboardInterrupt:
        print("\n🛑 User pressed Ctrl+C, exiting")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        # Ensure STT is completely stopped on exit
        stt.stop_recognition()
        print("✅ System stopped")

if __name__ == "__main__":
    main()