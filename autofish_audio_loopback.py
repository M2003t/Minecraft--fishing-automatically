import time
import numpy as np
import librosa
import pyautogui
import soundcard as sc

# ===== إعدادات عامة =====
TEMPLATE_MP3 = "fishing sound.mp3"  # اسم ملف القالب عندك
SR = 44100                           # معدل عيّنة موحّد
CHUNK_SEC = 1.2                      # طول نافذة الاستماع الحي
HOP_SEC = 0.1                        # تداخل زمني بين النوافذ
N_MFCC = 60                          # عدد معاملات MFCC
THRESHOLD = 0.80                      # عتبة التطابق (عدّل بين 0.82–0.92)
REFRACTORY_SEC = 2.5                 # مهلة بين عمليات الصيد (ثواني)

# ===== دوال مساعدة =====
def load_mono_resample_audio(path, sr=SR):
    y, sr_in = librosa.load(path, sr=None, mono=True)
    if sr_in != sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr)
    return y.astype(np.float32)

def mfcc_vector(y, sr=SR, n_mfcc=N_MFCC):
    # نقيس MFCC ونأخذ المتوسط عبر الزمن → متجه واحد
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    v = m.mean(axis=1)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))
# ===== حمّل القالب =====
print("[*] Loading template MP3...")
tmpl = load_mono_resample_audio(TEMPLATE_MP3, sr=SR)
# قصّ الصمت من الأطراف
tmpl, _ = librosa.effects.trim(tmpl, top_db=40)
tmpl_vec = mfcc_vector(tmpl, sr=SR)
print("[*] Template ready.")

# ===== تهيئة loopback من صوت النظام =====
spk = sc.default_speaker()  # مكبر الصوت الافتراضي في ويندوز
mic = sc.get_microphone(spk.id, include_loopback=True)  # loopback للصوت الخارج من الكمبيوتر

frame_len = int(CHUNK_SEC * SR)
hop_len = int(HOP_SEC * SR)
buf = np.zeros(frame_len, dtype=np.float32)
last_fire = 0.0

print("[*] Listening to system audio (loopback). Ctrl+C to stop.")
with mic.recorder(samplerate=SR, channels=2) as rec:
    while True:
        data = rec.record(numframes=hop_len)  # التقط جزء جديد
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)  # إلى mono
        # حرّك النافذة: احذف الأقدم وألصق الأحدث
        if len(data) < hop_len:
            continue
        buf = np.concatenate([buf[len(data):], data])

        # شِيل الصمت قبل القياس
        seg, _ = librosa.effects.trim(buf, top_db=35)
        if len(seg) < 0.2 * SR:
            # مقطع قصير/صامت
            continue

        vec = mfcc_vector(seg, sr=SR)
        sim = cosine_sim(vec, tmpl_vec)
        print(f"\rSimilarity: {sim:.3f}", end="")

        now = time.time()
        if sim >= THRESHOLD and (now - last_fire) > REFRACTORY_SEC:
            print(f"\n[!] Match {sim:.3f} → Right click (reel & cast)")
            # سحب ثم رمي (عدّل حسب طريقتك)
            pyautogui.click(button='right')
            time.sleep(0.25)
            pyautogui.click(button='right')
            last_fire = now

    