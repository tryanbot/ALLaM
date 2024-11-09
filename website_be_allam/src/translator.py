import requests
import re

class Translator():
    url = "https://nllb.infidea.dev/translate"
    def __init__(self, src_lang="eng_Latn", tgt_lang="arb_Arab"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    def __call__(self, text):
        return self.translate(text)
    def translate(self, text):
        reg = "[^\w\s.,]+"
        sep = re.findall(reg, text)
        spl = re.split(reg, text)
        res = self.translate_single(spl[0]) + " "
        for i, s in enumerate(spl[1:]):
            res+= sep[i] + " " + self.translate_single(s) + " "
        return res
    def translate_single(self, text):
        obj = {
        "text" : text,
        "src_lang" : self.src_lang,
        "tgt_lang" : self.tgt_lang
        }

        response = requests.post(self.url, json=obj)
        return response.json()['text']

class Translator_old():
    url = "https://nllb.infidea.dev/translate"
    def __init__(self, src_lang="eng_Latn", tgt_lang="arb_Arab"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    def __call__(self, text):
        return self.translate(text)
    def translate(self, text):
        obj = {
        "text" : text,
        "src_lang" : self.src_lang,
        "tgt_lang" : self.tgt_lang
        }

        response = requests.post(self.url, json=obj)
        return response.json()['text']