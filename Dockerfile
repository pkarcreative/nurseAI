FROM python:3.10
EXPOSE 8501
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install -r requirements.txt
ENTRYPOINT [ "streamlit","run" ]
CMD ["nurseAI_emotion_detection_app.py"]