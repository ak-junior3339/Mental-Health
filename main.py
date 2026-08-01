from fastapi import FastAPI,Request
import pickle 
import os
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 
app = FastAPI(title="Mental Health" , version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"model.pkl")
with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

# Defining the pydantic model(s)

class StudentData(BaseModel):
    age : int = Field(...,ge=10,le=100)
    gender : Literal['Male','Female']
    country : str
    academic_level : Literal['High School','Undergraduate','Graduate']
    most_used_platform : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours : float = Field(...,ge=0,le=24)
    daily_unlocks : int = Field(...,ge=0)
    study_hours : float = Field(...,ge=0,le=24)
    physical_activity_hours : float = Field(...,ge=0,le=24)
    sleep_hours_per_night : float = Field(...,ge=0,le=24)
    stress_level : Literal['Medium', 'Low', 'Very High', 'High']

class PredictionResponse(BaseModel) : 
    predicted_mental_health_score : float

@app.get("/health")
def health():
    return {"status": "ok"}

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get('/',response_class=HTMLResponse)
def greet(request: Request):
    return templates.TemplateResponse(
        request, 
        "index.html"
    )
