import ollama

def main():
    print("Select an option:")
    print("1: Chat example")
    print("2: Streaming example")
    print("3: Generate example")
    choice = input("Enter option number (1-3): ").strip()
    
    if choice == '1':
        # Simple chat example
        response = ollama.chat(
            model='llama3.2', 
            messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
        )
        print("Response:")
        print(response['message']['content'])
    elif choice == '2':
        # Streaming example
        print("Streaming response:")
        stream = ollama.chat(
            model='llama3.2', 
            messages=[{'role': 'user', 'content': 'Tell me a story in 3 sentences'}],
            stream=True
        )
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()  # newline after stream
    elif choice == '3':
        # Generate example
        response = ollama.generate(
            model='llama3.2', 
            prompt='Write a short poem about spring'
        )
        print("Generated response:")
        print(response['response'])
    else:
        print("Invalid option. Please run the script again and choose 1, 2, or 3.")

if __name__ == "__main__":
    # Pull the model (if not already pulled)
    ollama.pull(model='llama3.2')
    main()
