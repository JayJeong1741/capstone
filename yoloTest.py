import cv2
from ultralytics import YOLO
import socketio
import base64
import numpy as np

sio = socketio.Client()

@sio.event
def connect():
    print("✅ Socket.IO 서버에 연결됨")

# 서버 연결
sio.connect("http://192.168.35.139:3000")  # 본인의 Node.js 서버 주소로 변경

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 모델 로딩
model = YOLO("capstone1.0.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 웹캠에서 프레임을 읽을 수 없습니다.")
        break

    results = model.predict(source=frame, conf=0.6, iou=0.45)
    annotated_frame = results[0].plot()

    # === 추론 결과 출력 ===
    print("======================")
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            print(f"{result.names[class_id]},  {box.conf[0]:.2f}, {box.xyxy[0]}")
    print("======================")

    # FPS 표시
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # ==================== 프레임 전송을 위한 인코딩 ====================
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')  # base64 문자열로 인코딩
    sio.emit("frame", jpg_as_text)  # 서버로 전송

    # ==================== 디버깅용 화면 표시 ====================
    cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        sio.disconnect()
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()