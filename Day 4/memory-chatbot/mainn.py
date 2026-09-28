from ollama import chat

print("Synora - A Chatbot that remembers")

messages = []
messages.append({
    "role": "system",
    "content": "Answer a sentence in 50 words"
})

while True:
    question = input("you: ").strip()

    if question.lower() == "exit":
        print("Good Bye!")
        break

    elif question.lower() == "history":
        print("\n---- Chat history ----")
        for message in messages:
            if message["role"] == "user":
                print("You:", message["content"])
            elif message["role"] == "assistant":
                print("Bot:", message["content"])
        print("------------------\n")

    else:
        messages.append({
            "role": "user",
            "content": question
        })

        response = chat(
            model="llama3.2:latest",
            messages=messages
        )

        answer = response.message.content

        messages.append({
            "role": "assistant",
            "content": answer
        })

        print("bot:", answer)