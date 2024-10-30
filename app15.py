import streamlit as st

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("Video Object Detection App")

# 전체 레이아웃을 컨테이너로 감싸기
with st.container():
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

    # 파일 업로드
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    with col1:
        st.header("Original Video")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("Please upload a video file to display.")

    with col2:
        st.header("Object Detection Result Video")
        if "processed_video" in st.session_state:
            st.video(st.session_state["processed_video"])
        else:
            st.write("Object detection result will be displayed here.")

# 사물 검출 버튼 추가
if st.button("Run Object Detection"):
    if uploaded_file is not None:
        st.session_state["processed_video"] = uploaded_file
        st.success("Object detection completed and displayed on the right.")
    else:
        st.warning("Please upload a video file before running object detection.")
