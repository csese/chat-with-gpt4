#import openai
#audio_file= open("/path/to/file/audio.mp3", "rb")
#transcript = openai.Audio.transcribe("whisper-1", audio_file)



from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import sounddevice as sd
import numpy as np
import torch
import pyaudio


LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


def speech_to_text_hf(speech, sample_rate=16000):

    # Preprocess the audio
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    input_values = input_values.to(torch.float32)  # Convert input_values to float32

    # Get the model predictions
    with torch.no_grad():
        logits = model(input_values.input_values, attention_mask=input_values.attention_mask).logits


    # Convert the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    return predicted_sentences


def record_and_transcribe(chunk_size=1024, sample_rate=16000, channels=1, record_seconds=5):
    audio_format = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        frames = []
        for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

    finally:
        audio.terminate()

    # Convert the list of NumPy arrays to a single NumPy array
    speech = np.concatenate(frames, axis=0)

    transcription = speech_to_text_hf(speech, sample_rate=sample_rate)
    return transcription


#transcription = record_and_transcribe()
#print("Transcription:", transcription)


