import streamlit as st
from gtts import gTTS
from io import BytesIO

st.title("한국어 TTS 웹 애플리케이션")
st.write("텍스트를 입력하면 해당 텍스트를 음성으로 들려드립니다.")

# 텍스트 입력
text_input = st.text_area("변환할 텍스트를 입력하세요", "안녕하세요, TTS를 사용해보세요!")

# 버튼 클릭 시 음성 생성
if st.button("텍스트 음성 변환"):
    if text_input:
        # TTS 생성
        tts = gTTS(text=text_input, lang='ko')
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Streamlit에서 오디오 재생
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.write("텍스트를 입력해 주세요.")
