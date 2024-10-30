import streamlit as st

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("비디오 사물 검출 앱")

# 파일 업로드 버튼을 상단으로 이동
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        # 사물 검출 결과가 나타날 자리 확보 및 회색 박스 스타일 추가
        result_placeholder = st.empty()
        if "processed_video" in st.session_state and st.session_state["processed_video"] is not None:
            result_placeholder.video(st.session_state["processed_video"])
        else:
            result_placeholder.markdown(
                "<div style='border: 2px solid #d3d3d3; padding: 20px; text-align: center; color: #888;'>"
                "여기에 사물 검출 결과가 표시됩니다."
                "</div>",
                unsafe_allow_html=True,
            )

# 사물 검출 버튼 추가
if st.button("사물 검출 실행"):
    if uploaded_file is not None:
        # 여기에 사물 검출을 수행하는 코드를 추가하고, 결과를 st.session_state["processed_video"]에 저장
        # 예를 들어, 사물 검출 결과 영상을 processed_video_path에 저장했다고 가정하면,
        # st.session_state["processed_video"] = processed_video_path

        # 임시로 원본 파일을 사용하지 않고, 결과가 있을 때만 표시하도록 설정
        st.session_state["processed_video"] = None  # 실제 결과 영상으로 바꿔야 함
        result_placeholder.markdown(
            "<div style='border: 2px solid #d3d3d3; padding: 20px; text-align: center; color: #888;'>"
            "사물 검출 결과 영상이 여기에 표시됩니다."
            "</div>",
            unsafe_allow_html=True,
        )
        st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")
    else:
        st.warning("사물 검출을 실행하려면 비디오 파일을 업로드하세요.")
