import time
from pathlib import Path

import numpy as np
import librosa
import soundcard as sc
import pydirectinput


# ============================================================
# إعدادات عامة
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "fishing sound.mp3"

SR = 44100

# كل كم ثانية نقرأ جزءًا جديدًا من صوت النظام
READ_SEC = 0.05

# نحتفظ بصوت أطول قليلًا من القالب للبحث عن التطابق داخله
SEARCH_EXTRA_SEC = 0.50

# إعدادات البصمة الصوتية
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

# عتبة التطابق
THRESHOLD = 0.82

# مدة منع التكرار بعد اكتشاف الصوت
REFRACTORY_SEC = 2.5

# الفاصل بين سحب السنارة وإعادة رميها
RECAST_DELAY = 0.30

# مدة ضغط زر الفأرة فعليًا
RIGHT_CLICK_HOLD = 0.08


# ============================================================
# إعداد PyDirectInput
# ============================================================

pydirectinput.PAUSE = 0


# ============================================================
# تحميل الصوت
# ============================================================

def load_audio(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"\nلم أجد ملف الصوت:\n{path}\n"
        )

    print(f"[*] Template: {path}")

    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    # إزالة الصمت من البداية والنهاية
    y, _ = librosa.effects.trim(
        y,
        top_db=35
    )

    y = y.astype(np.float32)

    # إزالة DC offset
    y -= np.mean(y)

    # normalize
    peak = np.max(np.abs(y))

    if peak > 0:
        y /= peak

    return y


# ============================================================
# بصمة صوتية
# ============================================================

def audio_fingerprint(y: np.ndarray) -> np.ndarray:

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # إزالة اختلاف مستوى الصوت العام
    log_mel -= np.mean(
        log_mel,
        axis=0,
        keepdims=True
    )

    norm = np.linalg.norm(log_mel)

    if norm > 0:
        log_mel /= norm

    return log_mel.astype(np.float32)


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    a = a.ravel()
    b = b.ravel()

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# تحميل القالب
# ============================================================

print("[*] Loading fishing sound...")

template_audio = load_audio(
    TEMPLATE_FILE
)

template_fp = audio_fingerprint(
    template_audio
)

template_samples = len(
    template_audio
)

template_frames = template_fp.shape[1]

print(
    f"[*] Template duration: "
    f"{template_samples / SR:.3f} sec"
)

print(
    f"[*] Spectrogram size: "
    f"{template_fp.shape}"
)

print("[*] Template ready.")


# ============================================================
# Buffer للصوت الحي
# ============================================================

read_samples = int(
    READ_SEC * SR
)

buffer_samples = (
    template_samples
    + int(SEARCH_EXTRA_SEC * SR)
)

audio_buffer = np.zeros(
    buffer_samples,
    dtype=np.float32
)


# ============================================================
# البحث عن أفضل تطابق
# ============================================================

def find_best_match(audio: np.ndarray) -> float:

    live_fp = audio_fingerprint(
        audio
    )

    live_frames = live_fp.shape[1]

    if live_frames < template_frames:
        return 0.0

    best = -1.0

    for start in range(
        0,
        live_frames - template_frames + 1
    ):

        candidate = live_fp[
            :,
            start:start + template_frames
        ]

        score = cosine_similarity(
            candidate,
            template_fp
        )

        if score > best:
            best = score

    return best


# ============================================================
# كليك يمين
# ============================================================

def right_click():

    pydirectinput.mouseDown(
        button="right"
    )

    time.sleep(
        RIGHT_CLICK_HOLD
    )

    pydirectinput.mouseUp(
        button="right"
    )


# ============================================================
# عملية الصيد
# ============================================================

def fish():

    print("\n🎣 FISH!")

    # سحب السنارة
    right_click()

    time.sleep(
        RECAST_DELAY
    )

    # إعادة رمي السنارة
    right_click()


# ============================================================
# إعداد Loopback
# ============================================================

last_fire = 0.0

speaker = sc.default_speaker()

print(
    f"[*] Speaker: {speaker.name}"
)

loopback = sc.get_microphone(
    speaker.id,
    include_loopback=True
)

print("[*] Listening...")
print("[*] Ctrl+C to stop.\n")


# ============================================================
# Main loop
# ============================================================

try:

    with loopback.recorder(
        samplerate=SR,
        channels=2
    ) as recorder:

        while True:

            data = recorder.record(
                numframes=read_samples
            )

            # Stereo -> Mono
            if data.ndim == 2:

                data = np.mean(
                    data,
                    axis=1
                )

            data = data.astype(
                np.float32
            )

            if len(data) == 0:
                continue

            n = len(data)

            # تحديث rolling buffer
            if n >= buffer_samples:

                audio_buffer[:] = (
                    data[-buffer_samples:]
                )

            else:

                audio_buffer[:-n] = (
                    audio_buffer[n:]
                )

                audio_buffer[-n:] = data


            # قياس شدة الصوت
            rms = np.sqrt(
                np.mean(
                    audio_buffer ** 2
                )
            )

            # تجاهل الصمت
            if rms < 0.0005:
                continue


            # حساب التطابق
            score = find_best_match(
                audio_buffer
            )

            print(
                f"\rSimilarity: "
                f"{score:.3f}   ",
                end="",
                flush=True
            )


            now = time.monotonic()


            # ==================================================
            # اكتشاف صوت السمكة
            # ==================================================

            if (
                score >= THRESHOLD
                and
                now - last_fire
                >= REFRACTORY_SEC
            ):

                print(
                    f"\n[!] MATCH: "
                    f"{score:.3f}"
                )

                fish()

                last_fire = now


except KeyboardInterrupt:

    print(
        "\n[*] Stopped."
    )