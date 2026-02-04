from ollama_client import run_ollama_request
from detect import detect_text_in_image
from generate_text import tts_live


def main() -> None:
    print("image is processed")
    text = detect_text_in_image("images/IMG_5276.JPEG")
    print("llm is asked")
    prompt = "Simplify the follwing text into easier language and 4 sentences" + text
    result = run_ollama_request(prompt)
    print("tts is loading")
    tts_live(result)


if __name__ == "__main__":
    main()
