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
    ChatSession,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.traceback import install

# Enable pretty printing of exceptions with Rich
install()

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Create a console object for rich printing
console = Console()

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
    # print("Extracted filenames:", filenames)

    return filenames

model = GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])


async def ask_gemini_pro(question):
    """Ask Gemini Pro a question and print the response using ChatSession."""
    # Send the message to the chat session and get the response
    response = chat.send_message(question)

    # Print the response text
    for part in response.candidates[0].content.parts:
        console.print(part.text, style="bold green")


async def ask_gemini_pro_vision(question, source_folder, specific_file_name):
    """
    Ask Gemini Pro Vision a question about a specific image file.

    Args:
        question: The question to ask.
        source_folder: The folder containing the image file.
        specific_file_name: The name of the image file.
    """
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

    # Set the safety settings to block harmful content
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
            console.print(response.candidates[0].content.parts[0].text, style="bold green")
        else:
            console.print("No response candidates found.")


async def main():
    """Main function."""
    # Clear the console screen before displaying the welcome message
    os.system('cls' if os.name == 'nt' else 'clear')

    console.print(Markdown("# Welcome to Gemini Pro"), style="bold magenta")
    while True:
        user_input = Prompt.ask("\nAsk your question (or type 'exit' to quit)", default="exit")
        if user_input.lower() == 'exit':
            console.print("\nExiting the program.", style="bold red")
            break
        else:
            specific_file_names = extract_filename(user_input)
            # Add two blank lines after user input
            console.print("\n")
            if specific_file_names:
                # console.print("Calling ask_gemini_pro_vision", style="bold yellow")
                specific_file_name = specific_file_names[0]
                await ask_gemini_pro_vision(
                    user_input,
                    "workspace",
                    specific_file_name
                )
            else:
                # console.print("Calling ask_gemini_pro", style="bold yellow")
                await ask_gemini_pro(user_input)
            # Add one blank line after the model's response
            # console.print("\n")

if __name__ == "__main__":
    asyncio.run(main())
