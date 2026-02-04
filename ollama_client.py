from ollama import chat


def run_ollama_request(
    prompt: str,
    model: str = "gemma3:4b", # *! Sollte in der Doku erwähnt werden, das man das manuel selber installieren muss!*
) -> str:
    """
    Start an Ollama model runner, send a prompt using the Ollama Python client,
    return the response, and stop the runner.

    :param prompt: Prompt to send to the model
    :param model: Ollama model name (e.g., "gemma3:3b")
    """

    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )

    return response.message.content
