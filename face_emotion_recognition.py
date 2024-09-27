# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:16:02 2024

@author: pkarmakar
"""
from transformers import pipeline
from PIL import Image

def facial_emotion_detection(image_path):
    emotion_detector = pipeline(model="dima806/facial_emotions_image_detection")
    #pil_image = Image.open(image_path)
    emotion_prediction = emotion_detector(image_path)
    result = max(emotion_prediction, key=lambda x:x['score'])['label']
    #emotions: happy, surprise, neutral, sad, fear
    return result

def suggest_action_facial(emotion):
    if emotion == 'happy':
        return "The patient seems to be in a good mood. Continue with the current care plan."
    elif emotion == 'sad':
        return "The detected facial emotion is Sad. The patient appears to be sad. Consider offering emotional support or engaging in conversation."
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
        return "Monitoring the patient's facial emotional state."

