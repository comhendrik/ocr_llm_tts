from ollama_client import run_ollama_request
from generate_text import tts_live
from splitting import split_image_into_characters
from detect import reconstruct_text_from_images

MODEL_PATH = "model.keras"
IMAGE_PATH = "images/bio_text.png"
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
2. Summarize the content into **maximum 4 sentences**.
3. Do not add any introductory or concluding conversational filler (like "Here is the text"). This is the most important, only return one german sentence block
4. Do not use any points, just sentences
5. ONLY SIMPLIFY THE TEXT, DO NOT ADD ANY INFORMATION THAT IS NOT PROVIDED: ONLY RETURN A SIMPLIFIED VERSION OF THE TEXT IN MAXIMUM 4 SENTENCES

### INPUT TEXT
{text}

### OUTPUT should be in this language ({LANGUAGE})
"""
    result = run_ollama_request(prompt)
    print(result)

    print("STEP 4/4: TTS GENERATION")
    tts_live(result)


if __name__ == "__main__":
    main()
