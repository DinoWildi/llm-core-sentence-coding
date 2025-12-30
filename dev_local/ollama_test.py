import ollama
response = ollama.chat(model='llama3', messages=[ { 'role': 'user', 'content': 'why is the sky blue?'},])
print(response)
