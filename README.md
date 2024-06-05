# If-else-tts

Install packages
```bash
pip3 install -r requirements.txt
```

## 1. If else
```python
def response_weather(day: str, location: str):
    if day == "0":
        if location == "HCM":
            weather = "Hôm nay trời nắng, nhiệt độ ba mươi độ."
        elif location == "HN":
            weather = "Hôm nay trời mưa, nhiệt độ hai mươi lăm độ."
        else:
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    elif day == "1":
        if location == "HCM":
            weather = "Ngày mai trời nắng râm, nhiệt độ khoảng 'hai mươi tám' độ."
        elif location == "HN":
            weather = "Ngày mai trời mưa lớn, nhiệt độ khoảng 'hai mươi ba' độ."
        else:
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    elif day == "2":
        if location == "HCM":
            weather = "Trong ba ngày tới ở thành phố Hồ Chí Minh, có lúc nắng gắt lúc râm mát, nhiệt độ trung bình là 'ba mươi mốt' độ."
        elif location == "HN":
            weather = "Trong ba ngày tới ở Hà Nội, trời mưa nhiều, nhiệt độ trung bình là 'hai mươi tám' độ."
        else: 
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    
    return text2speech(weather)
```

## 2. TTS
```python
import torch
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

def text2speech(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    return Audio(output, rate=model.config.sampling_rate)
```

## 3. Streamlit app
```python
streamlit run app.py
```
