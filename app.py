import numpy as np
import streamlit as st
from constants import WHISPER_MODELS, language_dict
import streamlit as st
from utils import translate_to_english, detect_language, write, read, get_key
import whisperx as whisper
import json
import pandas as pd
from pydub import AudioSegment
import os

if "btn1" not in st.session_state:
    st.session_state["btn1"] = False
if "btn2" not in st.session_state:
    st.session_state["btn2"] = False


class ByteEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.hex()
        return json.JSONEncoder.default(self, obj)


def disable_btn2():
    st.session_state["btn2"] = True


def disable_btn1():
    st.session_state["btn1"] = True


st.set_page_config(page_title="Whisper-X", layout="wide")
import torch

if torch.cuda.is_available():
    device = "gpu"
else:
    device = "cpu"
input, output = st.columns(2, gap="medium")
with input:
    st.header("Input")
    audio_file = open("audio.wav", "rb")
    audio_bytes = audio_file.read()
    # st.markdown("""**sample audio**""", unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/wav")
    # st.markdown("""**your audio file**""", unsafe_allow_html=True)
    audio_uploaded = st.file_uploader(
        label="Upload your file",
        type=["mp3", "wav"],
        help="Your input file",
        # on_change=disable_btn2,
        # disabled=st.session_state["btn1"],
    )
    # text_json = st.file_uploader(
    #     label="Aligned JSON",
    #     type=["json"],
    #     help="Your aligned json file",
    #     # disabled=st.session_state["btn2"],
    #     # on_change=disable_btn1,
    # )
    text_json = None

    # st.markdown("""**model**""", unsafe_allow_html=True)
    model_name = st.selectbox(
        label="Choose your model",
        options=WHISPER_MODELS,
        help="Choose a Whisper model.",
    )
    model_name = "base" if model_name == "" else model_name
    # st.markdown("**transcription**", unsafe_allow_html=True)
    transcription = st.selectbox(
        "transcription",
        options=["plain text", "srt", "vtt", "ass", "tsv"],
        help="Choose the format for the transcription",
    )
    translate = st.checkbox(
        "translate", help="Translate the text to English when set to True"
    )
    language = st.selectbox(
        label="language",
        options=list(language_dict.keys()) + list(language_dict.values()),
        help="Translate the text to English when set to True",
    )
    patience = st.number_input(
        label="patience",
        step=0.01,
        value=1.0,
        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    temperature = st.number_input(
        label="temperature",
        step=0.01,
        value=1.0,
        help="temperature to use for sampling",
    )
    suppress_tokens = st.text_input(
        "suppress_tokens",
        value="-1",
        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    initial_prompt = st.text_area(
        label="initial_prompt",
        help="optional text to provide as a prompt for the first window.",
    )
    condition_on_previous_text = st.checkbox(
        "condition_on_previous_text",
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
    )
    temperature_increment_on_fallback = st.number_input(
        label="temperature_increment_on_fallback",
        step=0.01,
        value=0.2,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
    )
    compression_ratio_threshold = st.number_input(
        label="compression_ratio_threshold",
        value=2.4,
        step=0.01,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    logprob_threshold = st.number_input(
        label="logprob_threshold",
        value=-1.0,
        step=0.01,
        help="if the average log probability is lower than this value, treat the decoding as failed",
    )
    no_speech_threshold = st.number_input(
        label="no_speech_threshold",
        value=0.6,
        step=0.01,
        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]
    try:
        if len(temperature) == 0:
            st.error("Choose correct value for temperature")
    except:
        pass
    # st.write(temperature)
    submit = st.button("Submit", type="primary")
with output:
    st.header("Output")
    import uuid

    name = str(uuid.uuid1())
    if submit:
        if audio_uploaded is None:
            # st.audio(audio_bytes, format="audio/wav")
            audio_uploaded = audio_file
        if audio_uploaded is not None:
            if audio_uploaded.name.endswith(".wav"):
                temp = AudioSegment.from_wav(audio_uploaded)
                temp.export(f"{name}.wav")

            if audio_uploaded.name.endswith(".mp3"):
                temp = AudioSegment.from_wav(audio_uploaded)
                temp.export(f"{name}.wav")

            # audio_bytes = audio_uploaded.read()
            # st.audio(audio_bytes, format="audio/wav")
            if language == "":
                model = whisper.load_model(model_name)
                with st.spinner("Detecting language..."):
                    detection = detect_language(f"{name}.wav", model)
                    language = detection.get("detected_language")
                    del model
                    # st.write(language)
            if len(language) > 2:
                language = get_key(language)
            segments_pre = st.empty()
            segments_post = st.empty()
            segments_post_json = st.empty()
            segments_post2 = st.empty()
            trans = st.empty()
            lang = st.empty()
            if text_json is None:
                with st.spinner("Running ... "):
                    decode = {"suppress_tokens": suppress_tokens, "beam_size": 5}
                    model = whisper.load_model(model_name)
                    with st.container():
                        with st.spinner(f"Running with {model_name} model"):
                            result = model.transcribe(
                                f"{name}.wav",
                                language=language,
                                patience=patience,
                                initial_prompt=initial_prompt,
                                condition_on_previous_text=condition_on_previous_text,
                                temperature=temperature,
                                compression_ratio_threshold=compression_ratio_threshold,
                                logprob_threshold=logprob_threshold,
                                no_speech_threshold=no_speech_threshold,
                                **decode,
                            )

                if translate:
                    result = translate_to_english(result, json=False)
                with open("transcription.json", "w") as f:
                    json.dump(result["segments"], f, indent=4, cls=ByteEncoder)
                with st.spinner("Running alignment model ..."):
                    model_a, metadata = whisper.load_align_model(
                        language_code=result["language"], device=device
                    )
                    result_aligned = whisper.align(
                        result["segments"],
                        model_a,
                        metadata,
                        f"{name}.wav",
                        device=device,
                    )

            if text_json is not None:
                if translate:
                    result = translate_to_english(text_json, json=True)
                with st.spinner("Running alignment model ..."):
                    model_a, metadata = whisper.load_align_model(
                        language_code=language, device=device
                    )

                    result_aligned = whisper.align(
                        text_json, model_a, metadata, audio_uploaded.name, device
                    )

            if text_json is None:
                words_segments = result_aligned["word_segments"]
                write(
                    f"{name}.wav",
                    dtype=transcription,
                    result_aligned=result_aligned,
                )
                trans_text = read(f"{name}.wav", transcription)
                trans.text_area(
                    "transcription", trans_text, height=None, max_chars=None, key=None
                )
                segments_pre.text_area(
                    "Segments before alignment",
                    result["segments"],
                    height=None,
                    max_chars=None,
                    key=None,
                )
            segments_post.text_area(
                "Word Segments after alignment",
                result_aligned["word_segments"],
                height=None,
                max_chars=None,
                key=None,
            )
            with open("segments.json", "w", encoding="utf-8") as f:

                json.dump(result_aligned["word_segments"], f, indent=False)

            segments_post2.text_area(
                "Segments after alignment",
                result_aligned["segments"],
                height=None,
                max_chars=None,
                key=None,
            )
            lang.text_input(
                "detected language", language_dict.get(language), disabled=True
            )
            os.remove(f"{name}.wav")
