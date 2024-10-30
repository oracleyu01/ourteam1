import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# Restaurant-related questions and answers
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

# Create embeddings and dataframe
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Define chatbot function
def get_response(user_input):
    # Embed user input
    embedding = encoder.encode(user_input)
    
    # Find the most similar answer
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # Add to conversation history
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Set up the page configuration
st.set_page_config(page_title="Streamly 식당 챗봇", page_icon="🤖", layout="wide")

# Dark theme styling
st.markdown("""
    <style>
    /* 전체 페이지 배경을 검정색으로 설정 */
    .stApp {
        background-color: #0e1117;
        color: #d1d5db;
    }
    /* 사이드바 배경 및 텍스트 스타일 */
    .css-1d391kg, .css-1kyxreq, .css-18ni7ap, .sidebar .sidebar-content {
        background-color: #0e1117;
        color: #d1d5db;
    }
    /* 버튼 스타일 */
    .stButton > button {
        color: #ffffff;
        background-color: #1f2937;
        border-radius: 8px;
    }
    /* 텍스트 입력 필드 스타일 */
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: #d1d5db;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.image("image.png", width=150)
st.sidebar.title("Streamly Streamlit Assistant")
st.sidebar.radio("모드 선택:", ["최신 업데이트", "Streamly와 대화"])
st.sidebar.checkbox("기본 상호작용 보기", value=True)
st.sidebar.write("""
    **기본 상호작용**  
    - **Streamlit에 대해 묻기**: Streamlit의 최신 업데이트, 기능 또는 이슈에 대해 질문할 수 있습니다.  
    - **코드 검색**: '코드 예제', '구문', '사용 방법' 등의 키워드를 입력하여 관련 코드 스니펫을 찾을 수 있습니다.  
    - **업데이트 탐색**: '업데이트' 모드로 전환하여 최신 Streamlit 업데이트를 자세히 확인하세요.
""")

# Main chatbot interface
st.title("Streamly 식당 챗봇")
st.write("식당에 대해 궁금한 점을 물어보세요! 예: '영업시간이 어떻게 되나요?'")

user_input = st.text_input("질문을 입력하세요...", "")

if st.button("질문하기"):
    if user_input:
        get_response(user_input)
        user_input = ""  # Clear input field

# Display conversation history
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
