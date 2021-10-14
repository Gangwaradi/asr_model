import streamlit as st
import pyaudio
import wave
# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

def recorder(record_time = 5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = record_time
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

st.text('Adesh Kumar(@gangwaradi)\n14.10.2021')
#@st.cache
#def load_data():
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")


#model = load_data()

record_time = st.slider('Give approximate record timing (Seconds)', min_value=1, max_value=25, value=5, step=1)

if st.button('Record Audio'):
    recorder(record_time = record_time)

    if st.button('Recognise Audio'):
        files = ['output.wav']
        transcription = quartznet.transcribe(paths2audio_files=files)
        st.write('**Audio was recognised as: **' + transcription)

