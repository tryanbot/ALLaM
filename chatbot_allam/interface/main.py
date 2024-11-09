import gradio as gr
from dotenv import load_dotenv
load_dotenv()

import os
import requests



def predict(message, history):
    url = os.environ["ENG_URL"]
    obj = {
        "message" : message,
        "history" : []
    }
    if len(history) > 0:
        obj['history'] = [{"role" : m["role"], "content":m['content']} for m in history]
    response = requests.post(url, json=obj)
    return response.json()['content']

gr.ChatInterface(predict, type="messages").launch(server_name="0.0.0.0")