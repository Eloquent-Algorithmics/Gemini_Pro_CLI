"""
This is the gemini pro module.
"""
import os
import base64
import asyncio
import spacy
from spacy.matcher import Matcher
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_filename(text):
    """Extracts filenames from the text."""
    # Process the text with spaCy
    doc = nlp(text)

    # Define a pattern for matching filenames with extensions
    pattern = [
        {"TEXT": {"REGEX": "^[^\\s\\/]+\\.(jpg|png|mkv|mov|mp4|webm)$"}}
    ]

    # Add the pattern to the matcher
    matcher = Matcher(nlp.vocab)
    matcher.add("FILENAME", [pattern])

    # Apply the matcher to the doc
    matches = matcher(doc)

    # Extract and return the matched filenames
    filenames = []
    for _, start, end in matches:
        # The matched span
        span = doc[start:end]
        filenames.append(span.text)

    # Debugging: print the extracted filenames
    print("Extracted filenames:", filenames)

    return filenames


async def ask_gemini_pro(question):
    """Ask Gemini Pro a question and print the response."""
    model = GenerativeModel("gemini-pro")
    responses = model.generate_content(
        question,
        # Set the generation configuration
        generation_config={
            "max_output_tokens": 512,
            "temperature": 0.5,
            "top_p": 0.5,
            "top_k": 25,
        },
        # Set the safety settings to block all harmful content
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )

    for response in responses:
        # Debugging: Check if 'candidates' list is not empty
        if not response.candidates:
            print("No candidates found in the response.")
            continue

        # Debugging: Check if 'parts' list is not empty
        if not response.candidates[0].content.parts:
            print("No parts found in the candidate's content.")
            continue

        # If both lists are not empty, proceed to print the text
        print(response.candidates[0].content.parts[0].text)


async def ask_gemini_pro_vision(question, source_folder, specific_file_name):
    """Ask Gemini Pro Vision a question about a specific file."""
    # Read the image file as bytes and encode it with base64
    image_path = os.path.join(source_folder, specific_file_name)
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create a Part object with the image data
    image_part = Part.from_data(data=encoded_image, mime_type="image/jpeg")

    # Set up the generation configuration
    generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
    }

    # Set the safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:
            HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT:
            HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
            HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
            HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Create a GenerativeModel object for the Gemini Pro Vision model
    model = GenerativeModel("gemini-pro-vision")

    # Make the request and stream the responses
    responses = model.generate_content(
        [image_part, question],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Handle the responses
    for response in responses:
        if response.candidates:
            print(response.candidates[0].content.parts[0].text)
        else:
            print("No response candidates found.")


async def main():
    """Main function."""
    while True:
        user_input = input("Ask your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        else:
            # Extract the specific file name from the user input
            specific_file_names = extract_filename(user_input)

            # Debugging: print which function will be called
            if specific_file_names:
                print("Calling ask_gemini_pro_vision")
                specific_file_name = specific_file_names[0]
                await ask_gemini_pro_vision(
                    user_input,
                    "workspace",
                    specific_file_name
                )
            else:
                print("Calling ask_gemini_pro")
                await ask_gemini_pro(user_input)


if __name__ == "__main__":
    asyncio.run(main())
