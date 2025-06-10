import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import socketio
import base64
import threading
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

# Picamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)  # 해상도 감소
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# 전역 변수
running = True
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

# 오디오 캐싱
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
audio_cache = {
    'wait': pygame.mixer.Sound("wait.mp3"),
    'done': pygame.mixer.Sound("mp3/done.mp3"),
    'beep': pygame.mixer.Sound("mp3/beep.mp3"),
    'plz': pygame.mixer.Sound("mp3/plz.mp3")
}
print("🎵 pygame.mixer 및 오디오 파일 캐싱 완료")

def setTime(class_name):
    """오디오 재생 시퀀스 처리"""
    try:
        if class_name in ['guideDog', 'whiteCane']:
            audio_cache['wait'].play()
            time.sleep(12)
            audio_cache['done'].play()
            time.sleep(3)
            start_time = time.time()
            while time.time() - start_time < 20:
                audio_cache['beep'].play()
                time.sleep(7)
        elif class_name in ['crutches', 'wheelChair']:
            audio_cache['wait'].play()
            time.sleep(12)
            audio_cache['plz'].play()
            start_time = time.time()
            while time.time() - start_time < 25:
                audio_cache['plz'].play()
                time.sleep(10)
    except Exception as e:
        print(f"❌ 오디오 재생 에러: {e}")

def cleanup_states():
    """오래된 객체 상태 정리"""
    current_time = time.time()
    for obj_key in list(object_states.keys()):
        state = object_states[obj_key]
        if current_time - state['start_time'] > 300:  # 5분 이상된 항목 제거
            del object_states[obj_key]
            print(f"🗑️ {obj_key} 상태 정리")

def object_detection():
    """객체 탐지 및 상태 관리 함수"""
    global running, object_states, current_frame, last_sent_time, population, active_person_ids
    try:
        model = YOLO("model/capstone2.2_ncnn_model")
        frame_count = 0
        print("🔍 객체 탐지 시작...")

        while running:
            frame = picam2.capture_array()
            if frame is None:
                print("❌ 프레임 읽기 실패")
                break

            frame_count += 1
            if frame_count % 3 != 0:  # 프레임 스킵
                continue

            results = model.track(source=frame, conf=0.7, iou=0.45, persist=True)
            annotated_frame = results[0].plot()

            with frame_lock:
                current_frame = annotated_frame  # 얕은 복사

            current_objects = set()
            current_time = time.time()
            current_datetime = datetime.now()

            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    obj_id = int(box.id)
                    if class_name in target_classes:
                        current_objects.add((class_name, obj_id))

            cleanup_states()
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
                    print(f"🚀 {class_name} (ID: {obj_id}) 탐지 시작")

                if state['is_detected'] and not state['has_sent'] and state['class'] != 'person':
                    elapsed_time = current_time - state['start_time']
                    if elapsed_time >= detection_duration:
                        print(f"🚨 {class_name} (ID: {obj_id}) 3초 이상 탐지됨")
                        data = {"id": int(id), "cid": int(cid), "cls": class_name}
                        json_str = json.dumps(data)
                        if class_name in ['fallen', 'carAccident']:
                            sio.emit("emergency_detected", json_str)
                            state['has_sent'] = True
                        else:
                            threading.Thread(target=setTime, args=(class_name,), daemon=True).start()
                            state['has_sent'] = True

            for obj_key in list(object_states.keys()):
                class_name, obj_id = obj_key.split('_')
                obj_id = int(obj_id)
                if (class_name, obj_id) not in current_objects:
                    state = object_states[obj_key]
                    if state['is_detected']:
                        print(f"ℹ️ {class_name} (ID: {obj_id}) 탐지 중단")
                        del object_states[obj_key]

            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time if inference_time > 0 else 0
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imwrite("debug_frame.jpg", annotated_frame)
            cv2.imshow("Camera", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
                break

            time.sleep(0.01)

    except Exception as e:
        print(f"❌ 객체 탐지 에러: {e}")
    finally:
        cleanup_camera()

def manage_population(current_objects, current_datetime):
    """인구 수 관리 및 전송 함수"""
    global population, last_sent_time, active_person_ids
    for class_name, obj_id in current_objects:
        if class_name == 'person':
            if obj_id not in active_person_ids:
                active_person_ids[obj_id] = {'last_seen': current_datetime, 'count': 0}
            active_person_ids[obj_id]['last_seen'] = current_datetime
            active_person_ids[obj_id]['count'] += 1
            if active_person_ids[obj_id]['count'] == min_detections:
                population += 1
                print(f"👤 사람 (ID: {obj_id}) 안정적 탐지, population: {population}")

    expired_ids = [
        obj_id for obj_id, info in active_person_ids.items()
        if (current_datetime - info['last_seen']) > timedelta(seconds=10)
    ]
    for obj_id in expired_ids:
        del active_person_ids[obj_id]
        print(f"🗑️ 사람 (ID: {obj_id}) 탐지 만료")

    if current_datetime - last_sent_time >= population_window:
        send_traffic(population, current_datetime)
        population = 0
        last_sent_time = current_datetime
        print("📊 population 전송 완료")

def send_traffic(population, timestamp):
    """인구 수 데이터를 서버로 전송"""
    url = "http://localhost:8080/main/api/traffic"
    data = {
        "id": {"id": id, "cid": cid, "date": timestamp.isoformat()},
        "population": population
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"📤 인구 수 전송: status={response.status_code}")
    except Exception as e:
        print(f"❌ 인구 수 전송 에러: {e}")

def cleanup_camera():
    """카메라 리소스 해제"""
    global picam2
    if picam2 is not None:
        picam2.stop()
        picam2.close()
        print("📷 카메라 리소스 해제")
    cv2.destroyAllWindows()

@sio.event
def connect():
    print("✅ 서버 연결 성공")

@sio.event
def connection(sessionInfo):
    global sessionId, room
    sessionId = sessionInfo
    room = sessionId + idCid
    print(f"sessionId: {sessionId}, room: {room}")
    sio.emit("connectionSuccess", room)

@sio.event
def connect_error(data):
    print(f"❌ 연결 실패: {data}")

@sio.event
def disconnect():
    global running, room_states
    print("🔌 서버 연결 종료")
    running = False
    for room_id in list(room_states.keys()):
        room_states[room_id]["send_frames_enabled"] = False
    cleanup_camera()

@sio.on("connected")
def on_connected():
    print("🎉 서버로부터 'connected' 이벤트 수신")

@sio.on("videoCall")
def start_sending_frames(data):
    room_id = data
    if room_id not in room_states or not room_states[room_id]["send_frames_enabled"]:
        room_states[room_id] = {"send_frames_enabled": True, "thread": None}
        room_states[room_id]["thread"] = threading.Thread(
            target=send_frames, args=(room_id,), daemon=True
        )
        room_states[room_id]["thread"].start()
        print(f"📱 videoCall: {room_id}에서 프레임 전송 시작")

@sio.on("stopVideo")
def stop_sending_frames(data):
    if data in room_states:
        room_states[data]["send_frames_enabled"] = False
        print(f"🛑 stopVideo: {data}에서 프레임 전송 중단")

def send_frames(room_id):
    """프레임 전송 함수"""
    global running, current_frame
    while room_states.get(room_id, {}).get("send_frames_enabled", False):
        with frame_lock:
            frame = current_frame if current_frame is not None else None
        if frame is None:
            time.sleep(0.1)
            continue
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_data = base64.b64encode(buffer).decode("utf-8")
        sio.emit("frame", {"room_id": room_id, "data": frame_data})
        time.sleep(0.1)  # 10FPS로 전송

if __name__ == "__main__":
    try:
        threading.Thread(target=object_detection, daemon=True).start()
        print("🔄 서버 연결 시도...")
        sio.connect("http://localhost:3000")
        sio.emit("connectionForAlarm", cid)
        sio.wait()
    except KeyboardInterrupt:
        print("⚠️ 키보드 인터럽트 종료")
    except Exception as e:
        print(f"⚠️ 프로그램 오류: {e}")
    finally:
        running = False
        for room_id in list(room_states.keys()):
            room_states[room_id]["send_frames_enabled"] = False
        if sio.connected:
            sio.disconnect()
        cleanup_camera()
        pygame.mixer.quit()
        print("👋 프로그램 종료")