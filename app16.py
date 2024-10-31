import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# 전체 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목 설정
st.title("프로젝트 제목 사물 검출 앱")

# 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])


# 전체 레이아웃을 컨테이너로 감싸기
with st.container(): # with 절로 하나의 기능을 하는 코드를 묶어줌 (가독성 높이기)
    col1, col2 = st.columns(2)  # 열을 균등하게 분배하여 넓게 표시

    # 파일 업로드
    # uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

    with col1:
        st.header("원본 영상")   # col1 영역의 제목
        if uploaded_file is not None:  # 영상이 업로드가 되었다면
            st.video(uploaded_file)   # 영상 플레이해라 !
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:    
        st.header("사물 검출 결과 영상")  # col2 에 해당하는 영역의 제목
        result_placeholder = st.empty()  # 빈 영역 확보
        if "processed_video" in st.session_state: # 사물검출 완료된 비디오가 있으면
            st.video(st.session_state["processed_video"]) # 그 비디오를 플레이해라
        else: 
            #st.write("여기에 사물 검출 결과가 표시됩니다.")
             result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
             )

# 사물 검출 버튼 추가
if st.button("사물 검출 실행"):  # 사물검출 실행이라는 버튼을 누르면
    if uploaded_file is not None:  # 업로드된 영상이 있다면
        st.session_state["processed_video"] = uploaded_file # 검출된 영상을 사용
        st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.") # 이 메세지 출력
    else:
        st.warning("사물 검출을 실행하려면 비디오 파일을 업로드하세요.")

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행 및 디버깅
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        if len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 검출 결과가 없을 때 로그 출력
            st.write(f"Frame {frame_count}: No detections")

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # 결과 비디오를 st.session_state에 저장하여 스트림릿에 표시
    st.session_state["processed_video"] = output_path
    result_placeholder.video(output_path)
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")