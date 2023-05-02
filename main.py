
from chat import * 
from STT import *
from TTS import * 

def chat_with_gpt4():

    while True:
        print("Your turn to speak:")
        user_input = record_and_transcribe()
        if user_input is not None:
            print(f"User: {user_input}")

            # Generate response using GPT-4 model
            response = chat(user_input[0])

            # Remove input prompt from response
            gpt_answer = response

            print(f"GPT-4: {gpt_answer}")

            # Speak the response
            text_to_speech_live(gpt_answer)
        else:
            print("Could not understand the input.")

chat_with_gpt4()