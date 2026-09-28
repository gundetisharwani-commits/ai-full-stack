import ollama

response = ollama.chat(model="llama3.2", messages=[
    {"role":"user","content":"why we call prompt?in 120 words"}])
print(response.message.content)