import whisperx as whisper

from deep_translator import GoogleTranslator
import os
from whisperx.utils import write_vtt, write_srt, write_ass, write_tsv, write_txt


def detect_language(filename, model):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file=filename)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    return {"detected_language": max(probs, key=probs.get)}


def translate_to_english(transcription, json=False):
    if json:
        for text in transcription:
            text["text"] = GoogleTranslator(source="auto", target="en").translate(
                text["text"]
            )
    else:

        for text in transcription["segments"]:
            text["text"] = GoogleTranslator(source="auto", target="en").translate(
                text["text"]
            )
    return transcription


def write(filename, dtype, result_aligned):

    if dtype == "vtt":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".vtt"), "w"
        ) as vtt:
            write_vtt(result_aligned["segments"], file=vtt)
    if dtype == "srt":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".srt"), "w"
        ) as srt:
            write_srt(result_aligned["segments"], file=srt)
    if dtype == "ass":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".ass"), "w"
        ) as ass:
            write_ass(result_aligned["segments"], file=ass)
    if dtype == "tsv":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".tsv"), "w"
        ) as tsv:
            write_tsv(result_aligned["segments"], file=tsv)
    if dtype == "plain text":
        print("here")
        print(filename)
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".txt"), "w"
        ) as txt:
            write_txt(result_aligned["segments"], file=txt)


def read(filename, transc):
    if transc == "plain text":
        transc = "txt"
    filename = filename.split(".")[0]
    print(filename)
    with open(f"{filename}.{transc}", encoding="utf-8") as f:
        content = f.readlines()
    content = " ".join(z for z in content)
    return content


from constants import language_dict


def get_key(val):
    for key, value in language_dict.items():
        if val == value:
            return key
    return "Key not found"
