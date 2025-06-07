import os
from typing import Optional, List, Dict
import openai
import time

class LLMClient:
    """
    LLMClient is responsible for interacting with the LLM model.
    It provides a query() method to send user text to the model and return the assistant's reply.
    It also maintains a conversation history to keep track of the conversation context.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        openai.api_key = self.api_key

        self.model_name = model_name

        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def query(self, user_text: str) -> Optional[str]:
        """
        Send user text to the LLM model and return the assistant's reply.
        """
        # add new user text to the conversation history
        self.conversation_history.append({"role": "user", "content": user_text})

        try:
            # response = openai.ChatCompletion.create(
            #     model=self.model_name,
            #     messages=self.conversation_history,
            #     temperature=0.7,
            #     max_tokens=500,
            # )
            # assistant_reply = response.choices[0].message.content.strip()
            assistant_reply = "I'm sorry, I can't answer that question."
            # add assistant reply to the conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        except Exception as e:
            print(f"❌ Failed to call LLM: {e}")
            return None