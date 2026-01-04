import streamlit as st
from scipy.io.wavfile import write, read
import speech_recognition as sr
from transformers import pipeline
import numpy as np
import torch
import warnings
import os

# Suppress Hugging Face warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

SAMPLE_RATE = 16000
DURATION = 5

def load_models():
    st.info("Loading models... This may take a moment.")
    try:
        text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        audio_classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        return text_classifier, audio_classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
def capture_audio(duration=DURATION, samplerate=SAMPLE_RATE):

def text_analysis(audio_data, text_classifier):
    try:
        temp_wav_path = "temp_audio.wav"
        scaled_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        write(temp_wav_path, SAMPLE_RATE, scaled_data)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            st.write(f"**Text recognized:** '{text}'")
            text_result = text_classifier(text)
            return {item['label']: item['score'] for item in text_result[0]}
    except sr.UnknownValueError:
        st.warning("Text-based analysis failed: Could not understand audio. Try speaking more clearly.")
        return {}
    except sr.RequestError as e:
        st.warning(f"Text-based analysis failed: Could not request results from Google Speech Recognition service; {e}")
        return {}
    except Exception as e:
        st.warning(f"An error occurred during text analysis: {e}")
        return {}

def audio_analysis(audio_data, audio_classifier):
    try:
        audio_input = {"raw": audio_data, "sampling_rate": SAMPLE_RATE}
        audio_result = audio_classifier(audio_input)
        return {item['label']: item['score'] for item in audio_result}
    except Exception as e:
        st.warning(f"An error occurred during audio analysis: {e}")
        return {}

def combine_results(text_scores, audio_scores):
    combined_scores = {}
    common_emotions = set(text_scores.keys()).intersection(set(audio_scores.keys()))
    for emotion in common_emotions:
        combined_scores[emotion] = (text_scores.get(emotion, 0) + audio_scores.get(emotion, 0)) / 2
    for emotion, score in text_scores.items():
        if emotion not in common_emotions:
            combined_scores[emotion] = score
    for emotion, score in audio_scores.items():
        if emotion not in common_emotions:
            combined_scores[emotion] = score
    if combined_scores:
        final_emotion = max(combined_scores, key=combined_scores.get)
    else:
        final_emotion = "Undetermined"
    return combined_scores, final_emotion

def main():
    # --- Decorative UI ---
    st.set_page_config(page_title="🎤 Speech Emotion Detector", page_icon="🎶", layout="centered")
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        }
        .stApp {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #2d3142;
            text-align: center;
            margin-bottom: 0.5em;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #4f5d75;
            text-align: center;
            margin-bottom: 2em;
        }
        .emotion-box {
            background: #fff3e6;
            border-radius: 12px;
            padding: 1.5em;
            margin-bottom: 1em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-title">🎤 Speech Emotion Detection <span style="font-size:1.5rem;">(Text & Audio)</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a WAV file to transcribe and predict your emotion using both text and audio analysis.<br>Powered by <b>Hugging Face Transformers</b> and <b>Streamlit</b>.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn.pixabay.com/photo/2017/01/31/13/14/emoji-2025797_1280.png", width=120)
        st.markdown("""
        ## Instructions
        1. Upload a WAV file (16kHz, mono)
        2. Wait for the analysis results
        3. See your detected emotion!
        """)
        st.markdown("---")
        st.markdown("Made with ❤️ by Your Name")

    text_classifier, audio_classifier = load_models()
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        try:
            SAMPLE_RATE, audio_data = read(uploaded_file)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            st.audio(uploaded_file, format='audio/wav', sample_rate=SAMPLE_RATE)
            text_scores = text_analysis(audio_data, text_classifier)
            audio_scores = audio_analysis(audio_data, audio_classifier)
            if not text_scores and not audio_scores:
                st.error("Could not perform emotion analysis. Please check your file and try again.")
                return
            st.markdown('<div class="emotion-box"><b>Method 1: Text-based Analysis</b></div>', unsafe_allow_html=True)
            if text_scores:
                st.write({k: f"{v:.2f}" for k, v in sorted(text_scores.items(), key=lambda item: item[1], reverse=True)})
            else:
                st.write("Analysis failed.")
            st.markdown('<div class="emotion-box"><b>Method 2: Audio-based Analysis</b></div>', unsafe_allow_html=True)
            if audio_scores:
                st.write({k: f"{v:.2f}" for k, v in sorted(audio_scores.items(), key=lambda item: item[1], reverse=True)})
            else:
                st.write("Analysis failed.")
            combined_scores, final_emotion = combine_results(text_scores, audio_scores)
            st.markdown('<div class="emotion-box"><b>Final Combined Outcome</b></div>', unsafe_allow_html=True)
            st.success(f"Final Prediction: {final_emotion}")
            st.write({k: f"{v:.2f}" for k, v in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)})
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
