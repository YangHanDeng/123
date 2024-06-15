from modelscope.pipelines import pipeline
import wave
import numpy as np

ans = pipeline(
    'acoustic-noise-suppression',
    model='./speech_frcrn_ans_cirm_16k_official_result') # a folder that contains pytorch_model.bin and configuration.json

with open('testfile/noise1.wav', 'rb') as f:
    content = f.read()
print(type(content)) #bytes
result = ans(
    content,
    output_path='fileoutput.wav')

# 18s for 2min vid
#result= dict{'output_pcm', bytes array}
'''
with wave.open("myaudiofile.wav", "wb") as audiofile:
...     audiofile.setnchannels(1)
...     audiofile.setframerate(16000)
...     audiofile.setsampwidth(2)
...     audiofile.writeframes(result['output_pcm'])
'''