# =============================================================
# 🎬 Gerçek Zamanlı Video Sansür + Bip Overlay – TAM KOD
# =============================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
import whisper
import torch
import numpy as np
import threading
import queue
import time
from pydub import AudioSegment, generators
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except ImportError:
    from transformers import BertTokenizer, BertForSequenceClassification
    AutoTokenizer = BertTokenizer
    AutoModelForSequenceClassification = BertForSequenceClassification
import scipy.signal
from moviepy.editor import VideoFileClip, AudioFileClip
import base64
from io import BytesIO

# 🔧 Global Ayarlar
WHISPER_SR = 16000
THRESHOLD = 0.90
CHUNK_DURATION = 5
BEEP_FREQ = 1000
BEEP_GAIN = -6
CUSTOM_PROFANE = {"fenasik", "kerilim", "pompa", "fenasik kerilim", "donaltıp"}

# İş kuyrukları
processing_queue = queue.Queue()
results_queue = queue.Queue()
is_processing = False

@st.cache_resource(show_spinner=True)
def load_models():
    try:
        whisper_model = whisper.load_model("small")
        model_name = "Overfit-GM/distilbert-base-turkish-cased-offensive"
        
        # Tokenizer ve model yükleme
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            st.warning(f"AutoTokenizer yükleme hatası, fallback kullanılıyor: {e}")
            from transformers import BertTokenizer, BertForSequenceClassification
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
        
        device = 0 if torch.cuda.is_available() else -1
        offense_detector = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
        )
        return whisper_model, offense_detector
    except Exception as e:
        st.error(f"Model yükleme hatası: {str(e)}")
        raise e

def is_profane(word: str, offense_detector) -> bool:
    if word.lower() in CUSTOM_PROFANE:
        return True
    try:
        scores = offense_detector(word)[0]
        return next(s["score"] for s in scores if s["label"] == "PROFANITY") >= THRESHOLD
    except Exception:
        return False

def beep_over_intervals(audio_seg: AudioSegment, intervals_ms):
    output = audio_seg[:]
    for start_ms, end_ms in intervals_ms:
        duration_ms = max(0, int(end_ms - start_ms))
        if duration_ms <= 0:
            continue
        
        beep = generators.Sine(BEEP_FREQ).to_audio_segment(duration=duration_ms).apply_gain(BEEP_GAIN)
        before = output[:int(start_ms)]
        after = output[int(end_ms):]
        silence = AudioSegment.silent(duration=duration_ms)
        censored_part = silence.overlay(beep)
        output = before + censored_part + after
    return output

def extract_audio_chunks_from_video(video_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        duration = video.duration
        current = 0
        while current < duration:
            end = min(current + CHUNK_DURATION, duration)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio.subclip(current, end).write_audiofile(tmp.name, verbose=False, logger=None)
                yield {
                    "audio_path": tmp.name,
                    "chunk_id": int(current / CHUNK_DURATION),
                    "start_time": current,
                    "end_time": end,
                }
            current = end
        audio.close()
        video.close()
    except Exception as e:
        st.error(f"Video parçalara ayrılırken hata: {e}")
        return None

def process_audio_chunk(chunk_data, whisper_model, offense_detector):
    path = chunk_data["audio_path"]
    try:
        segment = AudioSegment.from_wav(path)
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)

        if segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples /= 32768.0

        if segment.frame_rate != WHISPER_SR:
            samples = scipy.signal.resample_poly(samples, WHISPER_SR, segment.frame_rate).astype(np.float32)

        result = whisper_model.transcribe(samples, language="tr", word_timestamps=True)

        words = []
        for seg in result["segments"]:
            for w in seg["words"]:
                start_ms = (w["start"] + chunk_data["start_time"]) * 1000
                end_ms = (w["end"] + chunk_data["start_time"]) * 1000
                words.append((w["word"].strip(), start_ms, end_ms))

        profanity_intervals = [(s, e) for w, s, e in words if is_profane(w, offense_detector)]
        censored_seg = beep_over_intervals(segment, [
            (s - chunk_data["start_time"] * 1000, e - chunk_data["start_time"] * 1000)
            for s, e in profanity_intervals
        ])

        masked_tokens = ["*" * len(w) if is_profane(w, offense_detector) else w for w, _, _ in words]
        masked_text = " ".join(masked_tokens)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            censored_seg.export(tmp_out.name, format="wav")
            censored_path = tmp_out.name

        os.unlink(path)
        return {
            **chunk_data,
            "text": result["text"].strip(),
            "masked_text": masked_text,
            "bad_words": {w for w, _, _ in words if is_profane(w, offense_detector)},
            "censored_audio_path": censored_path,
        }
    except Exception as e:
        if os.path.exists(path):
            os.unlink(path)
        return {
            **chunk_data,
            "text": f"[Hata: {e}]",
            "bad_words": set(),
            "masked_text": "",
            "censored_audio_path": None,
        }

def audio_processor_worker():
    global is_processing
    try:
        whisper_model, offense_detector = load_models()
        while is_processing:
            try:
                chunk_data = processing_queue.get(timeout=1)
                if chunk_data is None:
                    break
                res = process_audio_chunk(chunk_data, whisper_model, offense_detector)
                results_queue.put(res)
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Worker hatası: {e}")
    except Exception as e:
        st.error(f"Worker başlatma hatası: {e}")
    finally:
        is_processing = False

def combine_audio_segments(paths):
    combined = AudioSegment.empty()
    for p in paths:
        if p and os.path.exists(p):
            combined += AudioSegment.from_wav(p)
            os.unlink(p)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        combined.export(tmp.name, format="wav")
        return tmp.name

def play_audio_autoplay(wav_path, label="Sansürlü Parça"):
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        Tarayıcınız ses etiketini desteklemiyor.
    </audio>
    """
    st.markdown(f"**🎧 {label}**", unsafe_allow_html=True)
    st.markdown(audio_html, unsafe_allow_html=True)
    st.download_button(f"⬇ {label} indir", audio_bytes, file_name=f"{label}.wav", mime="audio/wav")

def main():
    st.set_page_config("🎬 Video Sansür + Bip", layout="wide")
    st.title("🎬 Gerçek Zamanlı Video Sansür (Bip Overlay)")
    st.markdown("Video sesindeki küfürlü kelimelerin tam üstüne 1 kHz bip yerleştirir, orijinal ses bozulmaz.")

    uploaded = st.file_uploader("Video dosyası seçin", ["mp4", "mkv", "mov", "avi", "webm"])
    if not uploaded:
        st.info("👈 Soldan bir video yükleyin.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded.name.split(".")[-1]) as tmp_vid:
        tmp_vid.write(uploaded.read())
        video_path = tmp_vid.name

    st.success("✅ Video yüklendi.")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.video(video_path)

    with right_col:
        start, stop = st.columns(2)
        with start:
            if st.button("🚀 Analizi Başlat", type="primary"):
                st.session_state.analysis = True
        with stop:
            if st.button("⏹ Durdur"):
                st.session_state.analysis = False

        transcript_area = st.empty()
        metric_area = st.empty()
        audio_stream_area = st.container()

    if st.session_state.get("analysis"):
        global is_processing
        if not is_processing:
            is_processing = True
            threading.Thread(target=audio_processor_worker, daemon=True).start()
            st.info("🔊 Ses analizine başlandı…")

            chunk_gen = extract_audio_chunks_from_video(video_path)
            total_bad_words = set()
            processed_chunks = 0

            for chunk in chunk_gen:
                if not st.session_state.get("analysis"):
                    break

                processing_queue.put(chunk)
                try:
                    res = results_queue.get(timeout=10)
                except queue.Empty:
                    continue

                processed_chunks += 1
                total_bad_words.update(res["bad_words"])

                with right_col:
                    transcript_area.write(res["masked_text"])
                    metric_area.metric("İşlenen Parça", processed_chunks)
                    with audio_stream_area:
                        play_audio_autoplay(res["censored_audio_path"], label=f"Parça_{processed_chunks}")

            is_processing = False
            processing_queue.put(None)

    with st.sidebar:
        st.markdown("""
        ## Nasıl çalışır?
        1. Video yükle
        2. *Analizi Başlat* – küfür anları tespit edilir, bip eklenir
        3. Her parça otomatik çalınır ve indirilebilir olur
        """)

if __name__ == "__main__":
    main()
