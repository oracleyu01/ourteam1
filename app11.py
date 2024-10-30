import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# Streamlit 설정
st.title("Batting Swing Detection")
st.write("업로드된 동영상에서 사물 검출을 수행합니다.")

# YOLO 모델 로드
model = YOLO('hitter_trained_model.pt')

# 클래스 이름 설정
class_names = ["geonchang", "other_class"]  # 수정 가능한 클래스 이름

# 파일 업로드
uploaded_file = st.file_uploader("동영상 파일을 업로드하세요", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Confidence 조정 슬라이더
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)

    # 임시 파일에 업로드된 동영상을 저장
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # 동영상 로드
    cap = cv2.VideoCapture(temp_video_path)

    # 동영상 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 임시 파일에 결과 동영상 저장
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_file.name, fourcc, fps, (frame_width, frame_height))

    # 프레임마다 YOLO 검출 수행
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 모델로 사물 검출 수행
        results = model(frame)

        for result in results:
            boxes = result.boxes  # 검출된 박스들
            for box in boxes:
                if box.conf >= confidence_threshold:  # 신뢰도 필터링
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    label = f"{class_names[class_id]} {confidence:.2f}"

                    # 검출 박스와 레이블 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 결과 프레임 저장
        out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()

    # Streamlit에 결과 동영상 표시
    st.video(output_temp_file.name)

    # 완료 후 임시 파일 삭제
    output_temp_file.close()
    st.success("검출이 완료되었습니다.")
