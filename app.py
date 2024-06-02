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


def response_weather(day: str, location: str):
    if day == 0:
        if location == "HCM":
            weather = "Thành phố Hồ Chí Minh hôm nay trời nắng, nhiệt độ ba mươi độ."
        elif location == "HN":
            weather = "Hôm nay trời mưa, nhiệt độ hai mươi lăm độ."
        else:
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    elif day == 1:
        if location == "HCM":
            weather = "Ngày mai trời nắng râm, nhiệt độ khoảng 'hai mươi tám' độ."
        elif location == "HN":
            weather = "Thành phố Hà Nội ngày mai trời mưa lớn, nhiệt độ khoảng 'hai mươi ba' độ."
        else:
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    elif day == 2:
        if location == "HCM":
            weather = "Trong ba ngày tới ở thành phố Hồ Chí Minh, có lúc nắng gắt lúc râm mát, nhiệt độ trung bình là 'ba mươi mốt' độ."
        elif location == "HN":
            weather = "Trong ba ngày tới ở Hà Nội, trời mưa nhiều, nhiệt độ trung bình là 'hai mươi tám' độ."
        else: 
            weather = "Xin lỗi, trung tâm không thể cung cấp thông tin về thành phố này."
    else:
        weather = f"Xin lỗi, trung tâm không thể cung cấp thông tin cho yêu cầu này. {day} - {location}"
    return weather


st.title("If-Else with Text-to-Speech (TTS)")
st.subheader("TTS model: [facebook/mms-tts-vie](https://huggingface.co/facebook/mms-tts-vie)")

st.header("Dự báo thời tiết") 

intro_1 = """Đây là trung tâm dự báo thời tiết của Việt Nam.
- Nhập không nếu bạn muốn biết thời tiết hiện tại.
- Nhập một nếu bạn muốn biết thời tiết trong một ngày tới.
- Nhập hai nếu bạn muốn biết thời tiết trong ba ngày tới.
"""

intro_2 = "Bạn muốn biết thời tiết ở đâu? Thành phố Hồ Chí Minh hay Hà Nội?"

waveform, rate = text2speech(intro_1, SPEED)
st.audio(waveform.numpy(), sample_rate=rate)
st.text(intro_1)

day = None
location = None
weather = None

day = st.number_input("Your input number is:", min_value=0, max_value=2, value=0)
if st.button("Submit day"):
    if day in [0, 1, 2]:
        waveform, rate = text2speech(intro_2, SPEED)
        st.audio(waveform.numpy(), sample_rate=rate)
        st.text(intro_2)
    else:
        st.warning("Day must be 0, 1, or 2.")


location = st.text_input("Your input location is (HCM, HN): ")
if st.button("Submit location"):
    weather = response_weather(day, location)

    if location in ["HCM", "HN"]:
        weather = response_weather(day, location)
        with st.spinner("Generating audio..."):
            waveform, rate = text2speech(weather, SPEED)
            st.audio(waveform.numpy(), sample_rate=rate)
            st.text(weather)

    else:
        st.warning("Location must be HCM or HN.")
