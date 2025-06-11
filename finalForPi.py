import cv2
from ultralytics import YOLO
import socketio
import base64
import threading
import platform
import time
import json
import requests
from datetime import datetime, timedelta
import pygame

# Socket.IO 클라이언트 인스턴스 생성
sio = socketio.Client()
id = "9"
cid = "26"
idCid = id + "&" + cid
sessionId = ""
room = ""

# 전역 변수
running = True
cap = None
object_states = {}
frame_lock = threading.Lock()
current_frame = None
room_states = {}
population = 0
last_sent_time = datetime.now()
detection_duration = 3
target_classes = ['guideDog', 'dog', 'fallen', 'whiteCane', 'carAccident', 'person', 'wheelChair', 'crutches', 'gudieWalker']
min_detections = 2
population_window = timedelta(seconds=120)
active_person_ids = {}
is_setTime_running = False  # setTime 실행 상태를 추적

# pygame 초기화
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
    print("🎵 pygame.mixer 초기화 완료 (frequency=44100, buffer=4096)")
except Exception as e:
    print(f"❌ pygame.mixer 초기화 에러: {e}")

def setTime(class_name):
    """오디오 재생 시퀀스 처리"""
    global is_setTime_running

    # setTime이 이미 실행 중이라면 새 호출을 무시
    if is_setTime_running:
        print(f"ℹ️ setTime({class_name}) 호출 무시: 이미 실행 중")
        return

    # setTime 실행 시작
    is_setTime_running = True
    try:
        if class_name == 'guideDog' or class_name == 'whiteCane':
            try:
                # wait.mp3 재생 (12초 대기)
                pygame.mixer.music.load("wait.mp3")  # 파일 경로 수정 필요
                pygame.mixer.music.play()
                print("▶️ wait.mp3 재생 중...")
                time.sleep(12)

                # done.mp3 재생 (4초 대기)
                pygame.mixer.music.load("mp3/done.mp3")  # 파일 경로 수정 필요
                pygame.mixer.music.play()
                print("▶️ done.mp3 재생 중...")
                time.sleep(3)

                # beep.mp3 30초 동안 반복 재생 (7초 파일 기준 약 4~5회)
                BEEP_DURATION = 7  # beep.mp3 길이 (초)
                PLAY_DURATION = 20  # 총 재생 시간 (초)
                start_time = time.time()
                play_count = 0

                pygame.mixer.music.load("mp3/beep.mp3")  # 파일 경로 수정 필요
                print("▶️ beep.mp3 재생 시작 (30초 동안 반복, 예상 횟수: ~4-5회)")

                while time.time() - start_time < PLAY_DURATION:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                        play_count += 1

                pygame.mixer.music.stop()
                print(f"⏹️ beep.mp3 재생 중지 (총 {play_count}회 반복)")
            except FileNotFoundError as e:
                print(f"❌ 오디오 파일 로드 에러: {e}")
            except Exception as e:
                print(f"❌ 오디오 재생 에러: {e}")
        elif class_name == 'crutches' or class_name == 'wheelChair':
            try:
                pygame.mixer.music.load("wait.mp3")  # 파일 경로 수정 필요
                pygame.mixer.music.play()
                print("▶️ wait.mp3 재생 중...")
                time.sleep(12)

                pygame.mixer.music.load("mp3/plz.mp3")  # 파일 경로 수정 필요
                print("▶️ plz.mp3 재생 중...")
                PLAY_DURATION = 25  # 총 재생 시간 (초)
                start_time = time.time()
                play_count = 0

                while time.time() - start_time < PLAY_DURATION:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                        play_count += 1
                        time.sleep(10)

                pygame.mixer.music.stop()
                print(f"⏹️ plz.mp3 재생 중지 (총 {play_count}회 반복)")
            except FileNotFoundError as e:
                print(f"❌ 오디오 파일 로드 에러: {e}")
            except Exception as e:
                print(f"❌ 오디오 재생 에러: {e}")
    finally:
        # setTime 실행 종료
        is_setTime_running = False
        print(f"✅ setTime({class_name}) 실행 완료")

def object_detection():
    """객체 탐지 및 상태 관리 함수"""
    global running, cap, object_states, current_frame, last_sent_time, population, active_person_ids

    try:
        cap = cv2.VideoCapture(0)  # 라즈베리파이 카메라 모듈 사용 시 경로 확인
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        try:
            model = YOLO("model/capstone2.5_ncnn_model")
        except Exception as e:
            print(f"❌ YOLO 모델 로드 에러: {e}")
            running = False
            return

        frame_count = 0
        print("🔍 객체 탐지 시작...")

        while running:
            ret, frame = cap.read()
            if not ret:
                print("❌ 웹캠에서 프레임을 읽을 수 없습니다.")
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            try:
                results = model.track(source=frame, conf=0.65, iou=0.45, persist=True)
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"❌ 객체 탐지 처리 에러: {e}")
                continue

            with frame_lock:
                current_frame = annotated_frame.copy()

            current_objects = set()
            current_time = time.time()
            current_datetime = datetime.now()

            print("======================")
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    obj_id = int(box.id)
                    print(f"{class_name} (ID: {obj_id}), {box.conf[0]:.2f}, {box.xyxy[0]}")
                    if class_name in target_classes:
                        current_objects.add((class_name, obj_id))
            print("======================")

            manage_population(current_objects, current_datetime)

            for class_name, obj_id in current_objects:
                obj_key = f"{class_name}_{obj_id}"
                if obj_key not in object_states:
                    object_states[obj_key] = {
                        'class': class_name,
                        'is_detected': False,
                        'start_time': 0,
                        'has_sent': False,
                        'count': 0
                    }

                state = object_states[obj_key]
                state['count'] += 1
                if state['count'] >= min_detections and not state['is_detected']:
                    state['start_time'] = current_time
                    state['is_detected'] = True
                    print(f"🚀 {class_name} (ID: {obj_id}) 탐지 시작, 시작 시간: {state['start_time']:.2f}")

                if state['is_detected'] and not state['has_sent'] and state['class'] != 'person':
                    elapsed_time = current_time - state['start_time']
                    print(f"⏱️ {class_name} (ID: {obj_id}) 경과 시간: {elapsed_time:.2f}초")
                    if elapsed_time >= detection_duration:
                        print(f"🚨 {class_name} (ID: {obj_id}) 3초 이상 탐지됨! 메시지 전송...")
                        data = {
                            "id": int(id),
                            "cid": int(cid),
                            "cls": class_name,
                        }
                        try:
                            json_str = json.dumps(data)
                            if class_name == 'fallen' or class_name == 'carAccident':
                                sio.emit("emergency_detected", json_str)
                                state['has_sent'] = True
                            else:
                                # 별도 스레드에서 오디오 재생
                                threading.Thread(target=setTime, args=(class_name,), daemon=True).start()
                                state['has_sent'] = True
                        except Exception as e:
                            print(f"❌ Socket.IO 메시지 전송 에러: {e}")

            for obj_key in list(object_states.keys()):
                class_name, obj_id = obj_key.split('_')
                obj_id = int(obj_id)
                if (class_name, obj_id) not in current_objects:
                    state = object_states[obj_key]
                    if state['is_detected']:
                        print(f"ℹ️ {class_name} (ID: {obj_id}) 탐지 중단, 상태 리셋")
                        del object_states[obj_key]

            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time if inference_time > 0 else 0
            text = f'FPS: {fps:.1f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            try:
                cv2.imwrite("debug_frame.jpg", annotated_frame)
            except Exception as e:
                print(f"❌ 디버그 프레임 저장 에러: {e}")

            # 라즈베리파이에서는 화면 출력 생략 가능
            if platform.system() != "Darwin" and platform.system() != "Linux":
                try:
                    cv2.imshow("Camera", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        running = False
                        break
                except Exception as e:
                    print(f"❌ 프레임 표시 에러: {e}")
                    break

            time.sleep(0.01)

    except Exception as e:
        print(f"❌ 객체 탐지 에러: {e}")
    finally:
        cleanup_camera()
        print("🔍 객체 탐지 종료")

def manage_population(current_objects, current_datetime):
    """인구 수 관리 및 전송 함수"""
    global population, last_sent_time, active_person_ids
    try:
        for class_name, obj_id in current_objects:
            if class_name == 'person':
                if obj_id not in active_person_ids:
                    active_person_ids[obj_id] = {'last_seen': current_datetime, 'count': 0}
                active_person_ids[obj_id]['last_seen'] = current_datetime
                active_person_ids[obj_id]['count'] += 1
                if active_person_ids[obj_id]['count'] == min_detections:
                    population += 1
                    print(f"👤 사람 (ID: {obj_id}) 안정적으로 탐지됨, population: {population}")

        expired_ids = [
            obj_id for obj_id, info in active_person_ids.items()
            if (current_datetime - info['last_seen']) > timedelta(seconds=10)
        ]
        for obj_id in expired_ids:
            del active_person_ids[obj_id]
            print(f"🗑️ 사람 (ID: {obj_id}) 탐지 만료, 제거됨")

        if current_datetime - last_sent_time >= population_window:
            send_traffic(population, current_datetime)
            population = 0
            last_sent_time = current_datetime
            print("📊 population 초기화 및 전송 완료")
    except Exception as e:
        print(f"❌ 인구 수 관리 에러: {e}")

def send_traffic(population, timestamp):
    """인구 수 데이터를 서버로 전송"""
    url = "http://118.218.212.147:59727/main/api/traffic"
    data = {
        "id": {
            "id": id,
            "cid": cid,
            "date": timestamp.isoformat()
        },
        "population": population
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"📤 인구 수 전송: status={response.status_code}, response={response.text}")
    except requests.RequestException as e:
        print(f"❌ 인구 수 전송 에러: {e}")

def cleanup_camera():
    """카메라 리소스 해제"""
    global cap
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            print("📷 카메라 리소스 해제 완료")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ 카메라 리소스 해제 에러: {e}")

@sio.event
def connect():
    print("✅ 서버에 연결됨")

@sio.event
def connection(sessionInfo):
    global sessionId, room
    try:
        sessionId = sessionInfo
        room = sessionId + idCid
        print(f"sessionId: {sessionId}, room: {room}")
        sio.emit("connectionSuccess", room)
    except Exception as e:
        print(f"❌ connection 이벤트 처리 에러: {e}")

@sio.event
def connect_error(data):
    print("❌ 연결 실패:", data)

@sio.event
def disconnect():
    global running, room_states
    print("🔌 서버 연결 종료됨")
    try:
        running = False
        for room_id in list(room_states.keys()):
            room_states[room_id]["send_frames_enabled"] = False
        cleanup_camera()
    except Exception as e:
        print(f"❌ disconnect 이벤트 처리 에러: {e}")

@sio.on("connected")
def on_connected():
    print("🎉 서버로부터 'connected' 이벤트 수신!")

@sio.on("videoCall")
def start_sending_frames(data):
    try:
        room_id = data
        print(f"room info: {room_id}")
        if room_id not in room_states or not room_states[room_id]["send_frames_enabled"]:
            room_states[room_id] = {"send_frames_enabled": True, "thread": None}
            room_states[room_id]["thread"] = threading.Thread(
                target=send_frames, args=(room_id,), daemon=True
            )
            room_states[room_id]["thread"].start()
            print(f"📹 videoCall 이벤트 시작: {room_id}에서 프레임 전송 시작")
    except Exception as e:
        print(f"❌ videoCall 이벤트 처리 에러: {e}")

@sio.on("stopVideo")
def stop_sending_frames(data):
    try:
        print(f"stopVideo: {data}")
        if data in room_states:
            room_states[data]["send_frames_enabled"] = False
            print(f"🛑 stopVideo 이벤트 수신: {data}에서 프레임 전송 중단")
    except Exception as e:
        print(f"❌ stopVideo 이벤트 처리 에러: {e}")

def send_frames(room_id):
    """프레임 전송 함수 (화질 낮춤)"""
    global running, current_frame
    try:
        while room_states.get(room_id, {}).get("send_frames_enabled", False):
            with frame_lock:
                frame = current_frame.copy() if current_frame is not None else None
            if frame is None:
                print(f"{room_id} 방에서 프레임 없음, 대기 중...")
                time.sleep(0.1)
                continue

            # 1. 해상도 줄이기 (예: 50%로 축소)
            scale_percent = 70  # 해상도 축소 비율
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # 2. JPEG 압축률 낮추기 (0~100, 낮을수록 저화질)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 50% 품질
            _, buffer = cv2.imencode(".jpg", frame_resized, encode_param)

            # 3. Base64로 인코딩
            frame_data = base64.b64encode(buffer).decode("utf-8")
            print(f"프레임 전송 시도: room_id={room_id}, 데이터 크기={len(frame_data)}")
            sio.emit("frame", {"room_id": room_id, "data": frame_data})
            time.sleep(0.03)
    except Exception as e:
        print(f"❌ 프레임 전송 에러: {e}")

if __name__ == "__main__":
    try:
        threading.Thread(target=object_detection, daemon=True).start()
        print("🔄 서버에 연결 중...")
        sio.connect("http://118.218.212.147:59726")
        sio.emit("connectionForAlarm", cid)
        sio.wait()
    except KeyboardInterrupt:
        print("⚠️ 키보드 인터럽트에 의한 종료")
    except socketio.exceptions.ConnectionError as e:
        print(f"❌ Socket.IO 연결 에러: {e}")
    except Exception as e:
        print(f"⚠️ 프로그램 오류: {e}")
    finally:
        running = False
        for room_id in list(room_states.keys()):
            room_states[room_id]["send_frames_enabled"] = False
        if sio.connected:
            try:
                sio.disconnect()
            except Exception as e:
                print(f"❌ Socket.IO 연결 해제 에러: {e}")
        cleanup_camera()
        try:
            pygame.mixer.quit()
            print("🎵 pygame.mixer 종료")
        except Exception as e:
            print(f"❌ pygame.mixer 종료 에러: {e}")
        print("👋 프로그램 종료")