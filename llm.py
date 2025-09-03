import json
from groq import Groq
from tts import TextToSpeech

system_prompts = {
    "sales": {
        "role": "system",
        "content": """"""
    }
}

class SalesBot:
    def __init__(self, api_key, tts):
        self.api_key = api_key
        self.system_prompt = system_prompts["sales"]
        self.groq_client = Groq(api_key=self.api_key)
        self.speaker = tts
        self.first_question_attempted = False
        self.conversation_history = []

    def set_prompt(self, prompt):
        self.system_prompt = {"role": "system", "content": prompt}

    def first_question(self):
        if not self.first_question_attempted:
            self.first_question_attempted = True
            first_line = self.extract_speech(self.system_prompt['content'])
            if first_line:
                self.speaker.text_to_audio_file(first_line)
                self.conversation_history.append({"role": "assistant", "content": first_line})

    def extract_speech(self, text):
        if "<speech>" in text and "</speech>" in text:
            return text.split("<speech>")[1].split("</speech>")[0].strip()
        return text

    def call_llm(self, messages):
        completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.7,
        )
        return completion.choices[0].message.content

    def ask_question(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        messages = [self.system_prompt] + self.conversation_history[-4:]
        response = self.call_llm(messages)
        reply_text = self.extract_speech(response)
        self.conversation_history.append({"role": "assistant", "content": reply_text})
        if reply_text:
            self.speaker.text_to_audio_file(reply_text)
