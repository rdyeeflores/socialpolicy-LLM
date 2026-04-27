from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Model choice
MODEL = "mistralai/mistral-small-3.1-24b-instruct"

# Conversation history
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful, concise assistant. "
            "Explain things clearly and avoid unnecessary jargon."
        ),
    }
]

print("Mistral Chat Demo")
print("Type 'exit' or 'quit' to stop.")
print()

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_reply = response.choices[0].message.content

    print()
    print("Mistral:", assistant_reply)
    print()

    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })