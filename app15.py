import streamlit as st

# 제목 설정
st.title("Video Object Detection App")

# 레이아웃 설정
col1, col2 = st.columns(2)

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
    # 이 부분은 사물 검출 후 생성된 비디오가 표시되는 자리입니다.
    # 초기에는 비어 있고, 검출 버튼 클릭 시 업데이트됩니다.
    if "processed_video" in st.session_state:
        st.video(st.session_state["processed_video"])
    else:
        st.write("Object detection result will be displayed here.")

# 사물 검출 버튼 추가
if st.button("Run Object Detection"):
    if uploaded_file is not None:
        # 사물 검출을 수행하는 함수(가상의 함수) 호출
        # 예를 들어, processed_video_path에 사물 검출된 비디오 경로 저장
        # 여기에서는 샘플로 기존 업로드된 비디오를 다시 사용해 표시
        st.session_state["processed_video"] = uploaded_file
        st.success("Object detection completed and displayed on the right.")
    else:
        st.warning("Please upload a video file before running object detection.")
