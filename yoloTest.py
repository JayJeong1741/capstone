import cv2
from ultralytics import YOLO
import socketio
import base64
import threading
import platform
import time

# Socket.IO 클라이언트 인스턴스 생성
sio = socketio.Client()
id = "2"
cid = "1"
idCid = id + "&" + cid

# 전역 변수 'running'을 명시적으로 선언
running = False
cap = None

# 프레임을 서버로 전송하는 함수 (스레드로 실행)
def send_frames():
    global running, cap

    try:
        # 웹캠 설정
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 낮춤
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # YOLO 모델 로딩
        model = YOLO("model/capstone2.5.pt")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # JPEG 품질 설정
        frame_count = 0  # 프레임 카운터

        print("🎥 비디오 스트리밍 시작...")

        while running:
            ret, frame = cap.read()
            if not ret:
                print("❌ 웹캠에서 프레임을 읽을 수 없습니다.")
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue  # 2프레임마다 1프레임만 처리

            # YOLO 모델 추론
            results = model.predict(source=frame, conf=0.7, iou=0.45)
            annotated_frame = results[0].plot()

            # 추론 결과 출력
            print("======================")
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    print(f"{result.names[class_id]}, {box.conf[0]:.2f}, {box.xyxy[0]}")
            print("======================")

            # FPS 계산 및 표시
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time if inference_time > 0 else 0
            text = f'FPS: {fps:.1f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 프레임 인코딩 및 전송
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # 서버로 프레임 전송
            if running:  # 전송 직전에 다시 한번 체크
                sio.emit("frame", {
                    "idCid": idCid,
                    "frame": jpg_as_text
                })

            # 디버깅용 저장 (macOS imshow 에러 회피)
            cv2.imwrite("debug_frame.jpg", annotated_frame)

            # 화면에 표시 (macOS에서는 생략)
            if platform.system() != "Darwin":
                cv2.imshow("Camera", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False
                    break

            # 스레드 응답성 향상을 위한 짧은 대기
            time.sleep(0.01)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        # 자원 정리
        cleanup_camera()
        print("🎥 비디오 스트리밍 종료됨")

def cleanup_camera():
    """카메라 리소스를 해제하는 함수"""
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        print("📷 카메라 리소스 해제 완료")
    cv2.destroyAllWindows()

# Socket.IO 연결 이벤트
@sio.event
def connect():
    print("✅ Socket.IO 서버에 연결됨")
    sio.emit("connection", idCid)

# 연결 실패 시
@sio.event
def connect_error(data):
    print("❌ 연결 실패:", data)

# 연결 종료 시
@sio.event
def disconnect():
    global running
    print("🔌 서버 연결 종료됨")
    running = False
    cleanup_camera()

@sio.on("connected")
def on_connected():
    print("🎉 서버로부터 'connected' 이벤트 수신!")

@sio.on("videoCall")
def sendFrame():
    global running
    if not running:  # running이 False일 때만 새 스레드 시작
        running = True
        threading.Thread(target=send_frames, daemon=True).start()
        print("📱 videoCall 이벤트 수신: 비디오 스트리밍 시작")

@sio.on("stopVideo")
def stopVideo():
    global running
    print("🛑 stopVideo 이벤트 수신: 비디오 스트리밍 중단")
    running = False

if __name__ == "__main__":
    try:
        # 서버 주소로 연결 (자신의 서버 주소로 변경)
        print("🔄 서버에 연결 중...")
        sio.connect("http://localhost:3000")
        sio.wait()
    except KeyboardInterrupt:
        print("⚠️ 키보드 인터럽트에 의한 종료")
    except Exception as e:
        print(f"⚠️ 프로그램 오류: {e}")
    finally:
        if running:
            running = False
        if sio.connected:
            sio.disconnect()
        cleanup_camera()
        print("👋 프로그램 종료")