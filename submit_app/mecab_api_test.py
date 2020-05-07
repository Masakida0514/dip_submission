import requests
import xml.etree.ElementTree as ET

target_url = 'https://jlp.yahooapis.jp/MAService/V1/parse'
client_id = 'dj00aiZpPWdKaWZIMDFZMklzdCZzPWNvbnN1bWVyc2VjcmV0Jng9ZWU-'

sentence = 'すもももももももものうち'

data = {
    "appid":client_id,
    "results":"ma",
    "sentence":sentence
}

response = requests.post(target_url, data=data)

root = ET.fromstring(response.text)

print(root)