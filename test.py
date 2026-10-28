import ollama

r = ollama.chat(model="gemma4", messages=[{"role": "user", "content": "Say hello in one sentence."}])
print(r["message"]["content"])