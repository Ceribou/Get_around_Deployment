import mlflow 
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile

description = """
API for Getaround predictions : this application may allow you to have an estimation on the rental price per day of your car, just by giving the applications some features regarding your car. 

## Machine-Learning 

Where you can:
* `/predict` the rental price of the car
* `/batch-predict` where you can upload a file to get prediction for the car


Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction",
        "description": "Introduction of the application."
    },
    {
        "name": "Predictions",
        "description": "Endpoints that uses our Machine Learning model for predicting rental price per day."
    }
]

app = FastAPI(
    title="🚗💨 GetAround API",
    description=description,
    version="0.1",
    contact={
        "name": "Cerise B."
    },
    openapi_tags=tags_metadata
)

class PredictionFeatures(BaseModel):
    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford', 'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors', 'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati', 'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT', 'Subaru', 'Suzuki', 'Toyota', 'Yamaha'] = "Citroën"
    mileage: Union[int, float] = 0
    engine_power: Union[int, float] = 0
    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro'] = "diesel"
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green'] = "black"
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van'] = "convertible"
    private_parking_available: Literal["yes", "no"] = "no"
    has_gps: Literal["yes", "no"] = "no"
    has_air_conditioning : Literal["yes", "no"] = "no"
    automatic_car : Literal["yes", "no"] = "no"
    has_getaround_connect : Literal["yes", "no"] = "no"
    has_speed_regulator : Literal["yes", "no"] = "no"
    winter_tires : Literal["yes", "no"] = "no"


@app.get("/", tags=["Introduction"])
async def index():
    message = "Welcome! This `/` is the most simple and default endpoint. If you want to have an estimation of the daily rental price of your car, check out documentation of the api at `/docs`"
    return message

@app.post("/predict", tags=["Predictions"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction for one car. Endpoint will return a dictionnary like this:

    ```
    {'prediction': Estimated rental price per day in euros}
    ```

    You need to give this endpoint all columns values (indicate the appropriate number or choose the right option among the list) as follows :
    ```
    - model_key: 'Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford', 'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors', 'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati', 'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT', 'Subaru', 'Suzuki', 'Toyota', 'Yamaha'.
    - mileage: nb of kilometers of the car
    - engine_power: engine power of the car
    - fuel: diesel, petrol, hybrid_petrol, electro
    - paint_color: 'black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green'
    - car_type: 'convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van'
    ```
    
    For the next criteria, please just indicate "yes" of "no" to each statement :
    ```
    - private_parking_available
    - has_gps
    - has_air_conditioning 
    - automatic_car
    - has_getaround_connect 
    - has_speed_regulator 
    - winter_tires
    ```
    """
    # Read data 
    df = pd.DataFrame(dict(predictionFeatures), index=[0])

    # Log model from mlflow 
    logged_model = 'runs:/1ba78b657fc4426c8bec1a2731fecab7/getaround_project'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    prediction = loaded_model.predict(df)

    # Format response
    response = {"prediction": f"{round(prediction.tolist()[0])} euros"}
    return response


@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Make prediction on a batch of observation. This endpoint accepts only **csv files** containing 
    all the trained columns WITHOUT the target variable. 
    """
    # Read file 
    df = pd.read_csv(file.file)

    # Log model from mlflow 
    logged_model = 'runs:/1ba78b657fc4426c8bec1a2731fecab7/getaround_project'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    predictions = loaded_model.predict(df)

    return predictions.tolist()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)