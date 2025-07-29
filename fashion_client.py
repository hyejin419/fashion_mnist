# gradio_client.py
import gradio as gr
import requests
import io

# 서버 URL (FastAPI)
API_URL = "http://127.0.0.1:8000/predict"  # ← 백엔드 URL

# 예측 함수
def classify_fashion_image(image):
    if image is None:
        return "이미지를 업로드하세요."

    # 이미지를 PNG 형식으로 바이너리 변환
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # 요청 전송
    response = requests.post(
        API_URL,
        files={"file": ("image.png", image_bytes, "image/png")}
    )

    if response.status_code == 200:
        result = response.json()
        return f"예측 결과: {result['class_name']} (클래스 {result['class_index']})"
    else:
        return f"오류 발생: {response.text}"

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=classify_fashion_image,
    inputs=gr.Image(type="pil", image_mode="L", height=28, width=28, label="28x28 흑백 이미지"),
    outputs=gr.Textbox(label="예측 결과"),
    title=" 패션 이미지 분류",
    description="Fashion MNIST 스타일의 28x28 흑백 이미지를 업로드하면 분류해드립니다."
)

if __name__ == "__main__":
    iface.launch()
