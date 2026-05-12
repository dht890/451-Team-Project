## How to run:
1. download zip and extract
2. Create virtual environment: ```python -m venv venv```
3. activate venv: ```venv\Scripts\activate```
5. install dependencies: ```pip install -r requirements.txt```
6. Environment variables: Create a `.env` file in the root directory and add:
```GEMINI_API_KEY="your_api_key_here"```
7. run: ```python dev_server.py```  
   (This wraps uvicorn reload with `venv` and `uploads` excluded from the file watcher. Plain `uvicorn main:app --reload` often spams reloads on Windows—especially under OneDrive—because `site-packages` files appear to change.)
8. follow the link ```http://127.0.0.1:8000```
