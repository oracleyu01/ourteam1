import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from PIL import Image
from moviepy.editor import VideoFileClip

# Streamlit 페이지 설정
st.set_page_config(page_title="Sophisticated Batting Swing Detection", page_icon="🎨")

# 스타일링 타이틀
st.markdown(
    "<h1 style='text-align: center; color: #D2691E;'>Sophisticated Batting Swing Detection</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #8B4513;'>Analyze batting swings with ease 🎬</h3>",
    unsafe_allow_html=True,
)

# 모나리자 이미지 표시
st.markdown("<h4 style='text-align: center;'>Sample Artwork</h4>", unsafe_allow_html=True)
monalisa_image = Image.open("monariza.png")  # 모나리자 이미지 경로를 실제 파일 경로로 수정하세요
st.image(monalisa_image, caption="Mona Lisa (by Leonardo da Vinci)", use_column_width=True)

# 모델 로드 함수
@st.cache_resource
def load_model():
    try:
        model = YOLO('hitter_trained_model.pt')  # 모델 파일 경로를 확인하세요
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()  # 모델이 없으면 앱 중지

# 클래스 이름 설정
class_names = ["geonchang", "other_class"]  # 수정 가능한 클래스 이름

# Sidebar 설정
st.sidebar.header("Settings ⚙️")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)

# 파일 업로드
uploaded_file = st.file_uploader("동영상 파일을 업로드하세요", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    st.sidebar.success("파일 업로드 완료")

    # 임시 파일에 업로드된 동영상을 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # 동영상 로드
    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 결과 동영상 저장
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_file.name, fourcc, fps, (frame_width, frame_height))

    st.write("🔄 **Processing video, please wait...**")
    
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
                if box.conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    label = f"{class_names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()

    # MoviePy를 사용하여 동영상을 웹 호환 형식으로 변환
    output_clip = VideoFileClip(output_temp_file.name)
    output_clip.write_videofile(output_temp_file.name, codec="libx264")

    # Streamlit에 결과 동영상 표시
    st.video(output_temp_file.name)

    # 완료 메시지
    st.success("🎉 검출이 완료되었습니다!")
else:
    st.write("동영상 파일을 업로드하세요.")
