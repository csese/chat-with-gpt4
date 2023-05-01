# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

import numpy as np
import simpleaudio as sa
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# Load the necessary models and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")



# Function to generate and play speech live
def text_to_speech_live(text):
    inputs = processor(text=text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    audio = speech.squeeze().numpy()

    # Normalize the audio by dividing it by the maximum absolute value
    audio_normalized = audio / np.max(np.abs(audio))

    # Convert the normalized audio to 16-bit PCM format
    audio_data = (audio_normalized * 32767).astype(np.int16)

    # Play the audio using the simpleaudio library
    play_obj = sa.play_buffer(audio_data, 1, 2, 16000)
    play_obj.wait_done()


text = "Hello, my dog is cute"
# Call the function with sample text
text_to_speech_live(text)

