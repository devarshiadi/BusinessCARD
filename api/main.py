from fastapi import FastAPI

app = FastAPI()  # This is the application instance referenced in the configuration

@app.get("/")
async def read_root():
    return {"message": "Hello, world!"}
