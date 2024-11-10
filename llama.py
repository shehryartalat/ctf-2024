from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import re
import time

FLAG = os.getenv("FLAG", "flag{fake_flag}")

TARGET = "sunday"

PROMPT = f"""Welcome to the llama jail!
Make our barely functional chatbot speak the word "{TARGET}" to get the flag.
Good luck :)"""

# Disable llama warnings
logging.basicConfig(level=logging.ERROR)

# Load tinyllama model - You don't need GPU to solve this challenge!
model = AutoModelForCausalLM.from_pretrained("nickypro/tinyllama-15M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nickypro/tinyllama-15M", legacy=True, trust_remote_code=True)


def jail(input_text):
    if len(input_text) > 5:
        print("Hey, I'm a tiny llama!")
        exit()

    if not re.match(r'^[a-z]{1,5}$', input_text):
        print("Hey, you are jailed!")
        exit()

    # Jailed...
    time.sleep(15)


def generate_response(input_text):
    tokenizer.pad_token = tokenizer.eos_token
    encoded_input = tokenizer.encode(input_text, padding=True, return_tensors="pt", truncation=True)

    # predict response based on input text
    generated = model.generate(encoded_input, pad_token_id=tokenizer.eos_token_id, max_new_tokens=20, num_beams=4)
    return tokenizer.decode(generated[0, len(encoded_input[0]):], skip_special_tokens=True)


def check_flag(response):
    if TARGET in response.lower().strip():
        return "Congrats! Here is your flag: " + FLAG
    return "Keep trying!"


if __name__ == "__main__":
    print(PROMPT)

    while True:
        try:
            input_text = input("> ").strip('\n')
            jail(input_text)
            response = generate_response(input_text)
            print(response)
            print(check_flag(response))
        except:
            exit()
