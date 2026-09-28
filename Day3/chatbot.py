import ollama
response=ollama.chat(
    model="llama3.2"
    messages=[
        {
            "role":"user",
            "content":"what is an AI"
        }
    ]
)
    print(response.message.content)

