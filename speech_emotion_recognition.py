# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:16:09 2024

@author: pkarmakar
"""
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from transformers import  Wav2Vec2FeatureExtractor, AutoConfig
import torch.nn.functional as F
import torch
import numpy as np
from pydub import AudioSegment


def predict_emotion_hubert(audio_file):
    """ inspired by an example from https://github.com/m3hrdadfi/soxan """


    model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion") # Downloading: 362M
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
    config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

    def speech_file_to_array(path, sampling_rate):
        # using torchaudio...
        # speech_array, _sampling_rate = torchaudio.load(path)
        # resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
        # speech = resampler(speech_array).squeeze().numpy()
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(sampling_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return sound_array

    sound_array = speech_file_to_array(audio_file, sampling_rate)
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to("cpu").float() for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{
        "emo": config.id2label[i],
        "score": round(score * 100, 1)}
        for i, score in enumerate(scores)
    ]
    return [row for row in sorted(outputs, key=lambda x:x["score"], reverse=True) if row['score'] != '0.0%'][:2]

def voice_emotion_detection(audio_file):

    result = predict_emotion_hubert(audio_file)
    result = result[0]['emo'].split("_")[-1]
    # emotions: angry, disgust, fear, happy, neutral, sad, surprise
    return result

def suggest_action_voice(emotion):
    if emotion == 'happy':
        return "The patient seems to be in a good mood. Continue with the current care plan."
    elif emotion == 'sad':
        return "The patient appears to be sad. Consider offering emotional support or engaging in conversation."
    elif emotion == 'angry':
        return "The patient seems frustrated. Check if they are uncomfortable or in need of assistance."
    elif emotion == 'fear':
        return "The patient is showing signs of fear and anxiety. Calm them down or check for any medical concerns."
    elif emotion == 'surprise':
        return "The patient is surprised. Ensure that they are not startled by any sudden changes."
    elif emotion == 'neutral':
        return "The patient appears neutral. Continue monitoring their emotional state."
    elif emotion == 'disgust':
        return "The patient seems disgusted. Check if something in the environment or care routine is causing discomfort."
    else:
        return "Monitoring the patient's voice emotional state."


