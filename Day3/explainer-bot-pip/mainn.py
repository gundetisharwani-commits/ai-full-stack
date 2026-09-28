import ollama

name = input("Enter your name: ")
# department = input("Enter your department: ")
# print(f"my name is {name} and I'm from {department} department.")
system='''
you explain programming error messages to a 2nd year emgineering student.reply in 3 parts;
1)what it means in plain english
2)why it happens
3)how to fix it
keep it under 120 words.
'''
while True:
    data=input("enter the error message or exit to stop:").strip()
    if not data:
        print("enter an error: ")
    elif data.lower()=="exit":
        break
    elif data.lower()=="help":
        print("Enter an error message,i find you a solution.")
    else:
        try: 
            response = ollama.chat(model='gemma3:1b', messages=[
            {
                'role': 'system',
                'content': system

            },
            {
                'role': 'user', 
                'content': data
                }
        ])
            print(response.message.content)
        except Exception as e:
            print(f"Couldn't reach response:{e}")
