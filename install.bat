@echo off

echo Creating Conda environment...
call conda create -n gemini -c conda-forge python=3.12 -y

echo Activating Conda environment...
call conda activate gemini

echo Installing Python requirements...
call pip install -r requirements.txt

echo Downloading SpaCy model...
call python -m spacy download en_core_web_sm

echo Installation completed.
pause