## How to run:
1. download zip and extract
2. Create virtual environment: ```python -m venv venv```
3. activate venv: ```venv\Scripts\activate```
5. install dependencies: ```pip install -r requirements.txt```
6. Environment variables: Create a `.env` file in the root directory and add:
```OPENAI_API_KEY="your_api_key_here"```
7. run: ```uvicorn main:app --reload```
8. follow the link ```http://127.0.0.1:8000```

if you get this message:
```
INFO:     Will watch for changes in these directories: ['C:\\Users\\david\\Downloads\\451-Team-Project-main']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [16848] using WatchFiles
ERROR:    Error loading ASGI app. Could not import module "main".
```
change directories with ```cd 451-Team-Project```
