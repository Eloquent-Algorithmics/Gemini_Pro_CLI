# Gemini Pro CLI

## Use the Google Vertex API to send request to the Gemini Pro and Gemini Pro Vision LLMs

### Google Cloud SDK login
```
gcloud init

gloud auth login
```

### Create a Conda Environment and install dependencies

```
.\install.bat
```

### Run Gemini Pro CLI

```
conda activate gemini

python -m gemini_pro
```

### Gemini Pro Vision

Add files to the /workspace folder and ask Gemini Pro to process them by filename.
```
Ask your question (or type 'exit' to quit):

What is the Year, Make, and, Model of the automobile in vehicle_1.jpg?

Extracted filenames: ['vehicle_1.jpg']
Calling ask_gemini_pro_vision

The year, make, and model of the automobile in vehicle_1.jpg is a 2021 Cadillac Escalade.
```
```
Ask your question (or type 'exit' to quit):

What is the subject of new_photo.jpg?

Extracted filenames: ['new_photo.jpg']
Calling ask_gemini_pro_vision

The subject of new_photo.jpg is Santa Claus.
```

