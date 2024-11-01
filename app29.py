import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO
import base64

# 기본 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 질문과 답변 데이터 설정
questions = [
    "영업시간이 어떻게 되나요?",
    "가격이 어떻게 되나요?",
    "주차가 가능한가요?",
    "사람들이 좋아하는 메뉴는 무엇인가요?",
    "목요일 12시에 예약 가능한가요?",
    "메뉴가 무엇이 있나요?",
    "위치가 어디인가요?"
]
answers = [
    "평일 영업시간은 오전 10시부터 오후 9시까지이며, 주말은 오전 10시부터 오후 10시까지입니다.",
    "비빔밥 9000원, 김치찌개 8000원, 된장찌개 8000원, 갈비탕 12000원입니다.",
    "네, 주차 가능합니다.",
    "가장 인기 있는 메뉴는 비빔밥과 갈비탕입니다.",
    "네, 목요일 오후 12시에 예약 가능합니다.",
    "메뉴에는 비빔밥, 김치찌개, 된장찌개, 갈비탕 등이 포함되어 있습니다.",
    "강남 국기원 사거리 삼원빌딩 1층에 위치해 있습니다."
]

# 질문 임베딩과 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력 저장
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 응답 함수
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산 후 가장 유사한 답변 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

    # gTTS를 사용하여 음성 생성
    tts = gTTS(text=answer['챗봇'], lang='ko')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # 오디오 자동 재생을 위해 base64로 변환
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# 페이지 설정
st.set_page_config(page_title="Streamly Chatbot", page_icon="🤖", layout="wide")

# 메인 챗봇 인터페이스
st.title("Streamly Restaurant Chatbot")

# 안내 문구
st.write("식당에 대해 궁금한 점을 물어보세요! 예: '영업시간이 어떻게 되나요?'")

user_input = st.text_input("Type your question here...", "")

# 질문에 대해 바로 소리로 답변
if st.button("Ask") and user_input:
    get_response(user_input)
    user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**User**: {message['user']}")
    st.write(f"**Chatbot**: {message['bot']}")
