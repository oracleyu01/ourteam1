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
    "평일 영업시간은 10:00 - 21:00, 주말은 10:00 - 22:00입니다.",
    "메뉴 가격은 비빔밥 9000원, 김치찌개 8000원, 된장찌개 8000원, 갈비탕 12000원입니다.",
    "네, 주차 가능합니다.",
    "사람들이 가장 좋아하는 메뉴는 비빔밥과 갈비탕입니다.",
    "목요일 12시에 예약 가능합니다.",
    "메뉴에는 비빔밥, 김치찌개, 된장찌개, 갈비탕 등이 있습니다.",
    "맛있는 한식당은 강남 국기원 사거리 삼원빌딩 1층에 있습니다."
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
st.set_page_config(page_title="Streamly 식당 챗봇", page_icon="🤖", layout="wide")

# 다크 테마 CSS
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #d1d5db;
    }
    .css-1kyxreq, .css-18ni7ap, .css-1d391kg {
        color: #d1d5db;
    }
    .sidebar .sidebar-content {
        background-color: #0e1117;
        color: #d1d5db;
    }
    .stButton > button {
        color: #ffffff;
        background-color: #1f2937;
        border-radius: 8px;
    }
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: #d1d5db;
    }
    </style>
    """, unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.image("/mnt/data/image.png", width=150)
st.sidebar.title("Streamly Streamlit Assistant")
st.sidebar.radio("모드 선택:", ["최신 업데이트", "Streamly와 대화"])
st.sidebar.checkbox("기본 상호작용 보기", value=True)
st.sidebar.write("""
    **기본 상호작용**  
    - **Streamlit에 대해 묻기**: Streamlit의 최신 업데이트, 기능 또는 이슈에 대해 질문할 수 있습니다.  
    - **코드 검색**: '코드 예제', '구문', '사용 방법' 등의 키워드를 입력하여 관련 코드 스니펫을 찾을 수 있습니다.  
    - **업데이트 탐색**: '업데이트' 모드로 전환하여 최신 Streamlit 업데이트를 자세히 확인하세요.
""")

# 메인 챗봇 인터페이스
st.title("Streamly 식당 챗봇")
st.write("식당에 대해 궁금한 점을 물어보세요! 예: '영업시간이 어떻게 되나요?'")

user_input = st.text_input("질문을 입력하세요...", "")

if st.button("질문하기"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
