python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install pygame numpy mutagen pydub websockets
python vmp.py --debug --music-dir "music" --backgrounds "backgrounds" 
