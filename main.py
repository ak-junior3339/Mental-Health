from fastapi import FastAPI,Request
import pickle 
import os
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 
import pandas as pd
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

@app.post("/predict", response_model=PredictionResponse)
async def predict_mental_health(data: StudentData):
    input_row = pd.DataFrame([{
        'Age'                       :data.age,
        'Gender'                    :data.gender,
        'Country'                   :data.country,
        'Academic_Level'            :data.academic_level,
        'Most_Used_Platform'        :data.most_used_platform,
        'Purpose_Of_Use'            :data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     :data.avg_daily_usage_hours,
        'Daily_Unlocks'             :data.daily_unlocks,
        'Study_Hours'               :data.study_hours,
        'Physical_Activity_Hours'   :data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     :data.sleep_hours_per_night,
        'Stress_Level'              :data.stress_level
    }])
    prediction = model.predict(input_row)[0] 
    return PredictionResponse(predicted_mental_health_score=round(float(prediction),2))
