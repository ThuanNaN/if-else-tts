import streamlit as st
import torch
from transformers import VitsModel, AutoTokenizer

SPEED = 120

@st.cache_resource
def load_model(model_name: str = "facebook/mms-tts-vie"):
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

@st.cache_data
def text2speech(text: str, speed: int):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    resample_rate = int(model.config.sampling_rate * (speed / 100))
    waveform = torch.nn.functional.interpolate(
        output[None, ...], scale_factor=(speed / 100), mode='linear')

    return waveform[0], resample_rate


def response_weather(request: int):
    if request == 0:
        weather = "Hôm nay trời nắng, nhiệt độ ba mươi độ."
    elif request == 1:
        weather = "Ngày mai trời mưa, nhiệt độ khoảng 'hai mươi lăm' độ."
    elif request == 2:
        weather = "Trong ba ngày tới, trời nắng gắt, nhiệt độ trung bình là 'ba mươi bốn' độ."
    else:
        weather = "Không rõ yêu cầu của bạn. Vui lòng nhập lại."
    return weather


st.title("Dự báo thời tiết")

intro = """Đây là trung tâm dự báo thời tiết của Việt Nam.
- Nhập không nếu bạn muốn biết thời tiết hiện tại.
- Nhập một nếu bạn muốn biết thời tiết trong một ngày tới.
- Nhập hai nếu bạn muốn biết thời tiết trong ba ngày tới.
"""
waveform, rate = text2speech(intro, SPEED)
st.text(intro)
st.audio(waveform.numpy(), sample_rate=rate)


number = None
number = st.number_input("Your input number is:",
                         min_value=0, max_value=2, value=0)

if st.button("Submit"):
    if number is not None:
        weather = response_weather(number)
        with st.spinner("Generating audio..."):
            waveform, rate = text2speech(weather, SPEED)
            st.audio(waveform.numpy(), sample_rate=rate)
            st.text(weather)
    else:
        st.warning("Please input a number before generating response.")
