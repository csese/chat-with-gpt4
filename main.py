from transformers import pipeline

from chat import * 

def chat_with_gpt4():
    user_input = speech_to_text()
    if user_input is not None:
        print(f"User: {user_input}")

        # Generate response using GPT-4 model
        prompt = f"{user_input}"
        gpt4_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        response = gpt4_generator(prompt, max_length=100)[0]['generated_text']
        
        # Remove input prompt from response
        response = response[len(prompt):].strip()
        print(f"GPT-4: {response}")

        # Speak the response
        text_to_speech(response)
    else:
        print("Could not understand the input.")

