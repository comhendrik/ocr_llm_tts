from ollama_client import run_ollama_request
from generate_text import tts_live
from splitting import split_image_into_characters
from detect import reconstruct_text_from_images

MODEL_PATH = "models/final_german_ocr_model_64px (9).keras"
IMAGE_PATH = "images/image5.png"
LANGUAGE = "German"

def main() -> None:

    print("STEP 1/4: IMAGE PREPROCESSING AND SPLITTING")
    images_array = split_image_into_characters(IMAGE_PATH, False)

    print("STEP 2/4: CONSTRUCT TEXT FROM IMAGES")
    text = reconstruct_text_from_images(images_array, MODEL_PATH)

    print("STEP 3/4: SIMPLIFYING TEXT WITH THE HELP OF AN LLM")
    prompt = f"""
### INSTRUCTIONS
Rewrite the text below into **{LANGUAGE}**.
1. Simplify the vocabulary and sentence structure to a **5th-grade reading level**.
2. Summarize the content into **strictly 4 sentences**.
3. Do not add any introductory or concluding conversational filler (like "Here is the text").

### INPUT TEXT
{text}

### OUTPUT ({LANGUAGE})
"""
    result = run_ollama_request(prompt)

    print("STEP 4/4: TTS GENERATION")
    tts_live(result)


if __name__ == "__main__":
    main()
