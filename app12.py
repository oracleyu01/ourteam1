import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
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

# 대화 이력을 저장하기 위한 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 응답 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산 후 가장 유사한 답변 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# 페이지 설정 및 다크 테마 스타일 적용
st.set_page_config(page_title="Streamly Restaurant Chatbot", page_icon="🤖", layout="wide")

# 다크 테마 CSS
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    header, [data-testid="stHeader"], .stApp > div:first-child {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    .stButton > button {
        color: #ffffff;
        background-color: #1f2937;
        border-radius: 8px;
    }
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# 사이드바 설정 (상단과 하단에 이미지 추가)
st.sidebar.image("image.png", use_column_width=True)
st.sidebar.title("Streamly Streamlit Assistant")
st.sidebar.radio("Select Mode:", ["Latest Updates", "Chat with Streamly"])
st.sidebar.checkbox("Show Basic Interactions", value=True)
st.sidebar.write("""
    **Basic Interactions**  
    - **Ask About Streamlit**: Ask about Streamlit's latest updates, features, or issues.  
    - **Search for Code**: Enter keywords like 'code example', 'syntax', 'how-to' to find related code snippets.  
    - **Navigate Updates**: Switch to 'Updates' mode to explore the latest Streamlit updates in detail.
""")
st.sidebar.image("image2.png", use_column_width=True)  # 사이드바 하단에 이미지 추가

# 메인 챗봇 인터페이스
st.title("Streamly Restaurant Chatbot")

# 안내 문구와 챗봇 이미지 한 줄에 배치
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("mini.png", width=30)  # 챗봇 이미지
with col2:
    st.write("식당에 대해 궁금한 점을 물어보세요! 예: '영업시간이 어떻게 되나요?")

user_input = st.text_input("Type your question here...", "")

if st.button("Ask"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시 (챗봇 응답에 이미지 아이콘 추가)
for message in st.session_state.history:
    st.write(f"**User**: {message['user']}")
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.image("mini.png", width=30)  # 챗봇 아이콘
    with col2:
        st.write(message['bot'])
