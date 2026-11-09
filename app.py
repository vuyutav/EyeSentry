import cv2
import time
import numpy as np
import mediapipe as mp
import threading
from flask import Flask, render_template, Response, jsonify
import tkinter as tk
from plyer import notification

app = Flask(__name__)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21  
ALARM_THRESHOLD = 2.0  

closed_eyes_start = None
is_drowsy = False  

start_work_time = time.time()
WORK_DURATION = 1 * 60   
REST_DURATION = 1 * 60   
rest_start_time = None
is_resting = False
is_too_close = False
remaining_rest_time = 0

last_distance_notif_time = 0

# Notifikasi Windows 
def kirim_notifikasi_jarak(judul, pesan):
    global last_distance_notif_time
    if time.time() - last_distance_notif_time > 15:
        try:
            notification.notify(
                title=judul,
                message=pesan,
                app_name='EyeSentry',
                timeout=5
            )
            last_distance_notif_time = time.time()
        except Exception as e:
            print("Gagal mengirim notifikasi Windows:", e)

# Pop-up Desktop Tkinter (Khusus Waktu Istirahat dengan Hitung Mundur Presisi)
def tampilkan_popup_istirahat():
    def run_popup():
        root = tk.Tk()
        root.title("EyeSentry - Waktunya Istirahat!")
        root.attributes('-topmost', True) 
        
        # Menghilangkan bilah judul (title bar) beserta tombol minimize/maximize/close (X)
        root.overrideredirect(True)
        
        root.geometry("500x320")
        root.configure(bg="#0f172a")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width / 2) - (500 / 2)
        y = (screen_height / 2) - (320 / 2)
        root.geometry(f'+{int(x)}+{int(y)}')

        label_judul = tk.Label(root, text=" Waktunya Istirahat Sejenak!", fg="#38bdf8", bg="#0f172a", font=("Arial", 18, "bold"))
        label_judul.pack(pady=25)

        label_desc = tk.Label(root, text="Istirahatkan mata Anda dari layar laptop.\nJendela akan menutup otomatis setelah waktu selesai.", fg="#cbd5e1", bg="#0f172a", font=("Arial", 11))
        label_desc.pack(pady=5)

       # Label Angka Hitung Mundur
        label_timer = tk.Label(root, text=f"{remaining_rest_time}s", fg="#fACC15", bg="#0f172a", font=("Arial", 36, "bold"))
        label_timer.pack(pady=15)
        # Fungsi Update Timer di Pop-up
        def update_popup_timer():
            global remaining_rest_time, is_resting
            if is_resting:
                if remaining_rest_time <= 1:
                    root.destroy()
                    return
                label_timer.config(text=f"{remaining_rest_time}s")
            
            root.after(1000, update_popup_timer)

        root.after(1000, update_popup_timer)
        root.mainloop()

    threading.Thread(target=run_popup, daemon=True).start()

def hitung_ear(landmarks, eye_indices, width, height):
    pts = np.array([(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def normalkan_pencahayaan(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def generate_frames():
    global closed_eyes_start, is_drowsy, start_work_time, rest_start_time, is_resting, is_too_close, remaining_rest_time
    camera = cv2.VideoCapture(0)
    popup_triggered = False

    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        h, w, _ = frame.shape

        # Logika Timer Istirahat
        if not is_resting:
            elapsed_work = current_time - start_work_time
            if elapsed_work >= WORK_DURATION:
                is_resting = True
                rest_start_time = current_time
                popup_triggered = False
        else:
            elapsed_rest = current_time - rest_start_time
            remaining_rest_time = int(REST_DURATION - elapsed_rest)
            
            # Panggil pop-up satu kali saat masuk mode istirahat
            if not popup_triggered:
                tampilkan_popup_istirahat()
                popup_triggered = True

            if remaining_rest_time <= 0:
                is_resting = False
                start_work_time = time.time()
                remaining_rest_time = 0
                popup_triggered = False

        # Proses Deteksi Wajah & Jarak
        enhanced_frame = normalkan_pencahayaan(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "Fokus / Terjaga"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Cek Jarak Terlalu Dekat berdasarkan lebar wajah
                x_coords = [int(pt.x * w) for pt in landmarks]
                face_width = max(x_coords) - min(x_coords)
                MAX_FACE_WIDTH = 320

                if face_width > MAX_FACE_WIDTH:
                    is_too_close = True
                    status_text = "PERINGATAN: Posisi Terlalu Dekat!"
                    color = (0, 165, 255)
                    is_drowsy = False
                    closed_eyes_start = None
                    kirim_notifikasi_jarak("EyeSentry: Peringatan Jarak!", "Posisi Anda terlalu dekat dengan layar!")
                else:
                    is_too_close = False
                    left_ear = hitung_ear(landmarks, LEFT_EYE, w, h)
                    right_ear = hitung_ear(landmarks, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0

                    for idx in LEFT_EYE + RIGHT_EYE:
                        pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                        cv2.circle(frame, pt, 1, (255, 255, 0), -1)

                    if avg_ear < EAR_THRESHOLD:
                        if closed_eyes_start is None:
                            closed_eyes_start = time.time()
                        
                        elapsed = time.time() - closed_eyes_start
                        if elapsed >= ALARM_THRESHOLD:
                            status_text = f"PERINGATAN: KANTUK! (EAR: {avg_ear:.2f})"
                            color = (0, 0, 255)
                            is_drowsy = True
                        else:
                            status_text = f"Mata Terpejam... (EAR: {avg_ear:.2f})"
                            color = (0, 255, 255)
                            is_drowsy = False
                    else:
                        closed_eyes_start = None
                        status_text = f"Fokus / Terjaga (EAR: {avg_ear:.2f})"
                        color = (0, 255, 0)
                        is_drowsy = False
        else:
            status_text = "Tidak Ada Siswa Terdeteksi"
            color = (0, 165, 255)
            closed_eyes_start = None
            is_drowsy = False
            is_too_close = False

        cv2.putText(frame, f"SmartEye: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global is_drowsy, is_resting, is_too_close, remaining_rest_time
    return jsonify({
        'drowsy': is_drowsy,
        'resting': is_resting,
        'too_close': is_too_close,
        'remaining_rest': remaining_rest_time
    })

if __name__ == '__main__':
    app.run(debug=True)