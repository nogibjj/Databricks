import mlflow
import pandas as pd
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
"""class tweet for sentiment analysis"""


class Tweet(BaseModel):
    text: str


# Method for prediction
def predict(text):
    """return prediction"""
    print(f"Accepted payload: {text}")
    my_data = {
        "selected_text": {0: text},
        "text": {0: text},
    }
    data = pd.DataFrame(data=my_data)
    result = loaded_model.predict(pd.DataFrame(data))
    return result


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model("model")
app = FastAPI()

# decoration for prediction
@app.post("/predict")
async def predict_tweet(tweet: Tweet):
    print(f"predict_tweet accepted json payload: {tweet}")
    result = predict(tweet.text)
    print(f"The result is the following payload: {result}")
    payload = {"TweetPosNeg": result.tolist()[0]}
    json_compatible_item_data = jsonable_encoder(payload)
    return JSONResponse(content=json_compatible_item_data)


# Root decoration
@app.get("/")
async def root():
    return {"message": "Hello Model"}


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
