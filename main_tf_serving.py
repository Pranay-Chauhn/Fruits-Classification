from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoints = "http://localhost:8501/v1/models/fruits_model:predict"

CLASS_NAMES = ['Apple 6',
               'Apple Braeburn 1',
               'Apple Crimson Snow 1',
               'Apple Golden 1',
               'Apple Golden 2',
               'Apple Golden 3',
               'Apple Granny Smith 1',
               'Apple Pink Lady 1',
               'Apple Red 1',
               'Apple Red 2',
               'Apple Red 3',
               'Apple Red Delicious 1',
               'Apple Red Yellow 1',
               'Apple Red Yellow 2',
               'Apple hit 1',
               'Apricot 1',
               'Avocado 1',
               'Avocado ripe 1',
               'Banana 1',
               'Banana Lady Finger 1',
               'Banana Red 1',
               'Beetroot 1',
               'Blueberry 1',
               'Cabbage white 1',
               'Cactus fruit 1',
               'Cantaloupe 1',
               'Cantaloupe 2',
               'Carambula 1',
               'Carrot 1',
               'Cauliflower 1',
               'Cherry 1',
               'Cherry 2',
               'Cherry Rainier 1',
               'Cherry Wax Black 1',
               'Cherry Wax Red 1',
               'Cherry Wax Yellow 1',
               'Chestnut 1',
               'Clementine 1',
               'Cocos 1',
               'Corn 1',
               'Corn Husk 1',
               'Cucumber 1',
               'Cucumber 3',
               'Cucumber Ripe 1',
               'Cucumber Ripe 2',
               'Dates 1',
               'Eggplant 1',
               'Eggplant long 1',
               'Fig 1',
               'Ginger Root 1',
               'Granadilla 1',
               'Grape Blue 1',
               'Grape Pink 1',
               'Grape White 1',
               'Grape White 2',
               'Grape White 3',
               'Grape White 4',
               'Grapefruit Pink 1',
               'Grapefruit White 1',
               'Guava 1',
               'Hazelnut 1',
               'Huckleberry 1',
               'Kaki 1',
               'Kiwi 1',
               'Kohlrabi 1',
               'Kumquats 1',
               'Lemon 1',
               'Lemon Meyer 1',
               'Limes 1',
               'Lychee 1',
               'Mandarine 1',
               'Mango 1',
               'Mango Red 1',
               'Mangostan 1',
               'Maracuja 1',
               'Melon Piel de Sapo 1',
               'Mulberry 1',
               'Nectarine 1',
               'Nectarine Flat 1',
               'Nut Forest 1',
               'Nut Pecan 1',
               'Onion Red 1',
               'Onion Red Peeled 1',
               'Onion White 1',
               'Orange 1',
               'Papaya 1',
               'Passion Fruit 1',
               'Peach 1',
               'Peach 2',
               'Peach Flat 1',
               'Pear 1',
               'Pear 2',
               'Pear 3',
               'Pear Abate 1',
               'Pear Forelle 1',
               'Pear Kaiser 1',
               'Pear Monster 1',
               'Pear Red 1',
               'Pear Stone 1',
               'Pear Williams 1',
               'Pepino 1',
               'Pepper Green 1',
               'Pepper Orange 1',
               'Pepper Red 1',
               'Pepper Yellow 1',
               'Physalis 1',
               'Physalis with Husk 1',
               'Pineapple 1',
               'Pineapple Mini 1',
               'Pitahaya Red 1',
               'Plum 1',
               'Plum 2',
               'Plum 3',
               'Pomegranate 1',
               'Pomelo Sweetie 1',
               'Potato Red 1',
               'Potato Red Washed 1',
               'Potato Sweet 1',
               'Potato White 1',
               'Quince 1',
               'Rambutan 1',
               'Raspberry 1',
               'Redcurrant 1',
               'Salak 1',
               'Strawberry 1',
               'Strawberry Wedge 1',
               'Tamarillo 1',
               'Tangelo 1',
               'Tomato 1',
               'Tomato 2',
               'Tomato 3',
               'Tomato 4',
               'Tomato Cherry Red 1',
               'Tomato Heart 1',
               'Tomato Maroon 1',
               'Tomato Yellow 1',
               'Tomato not Ripened 1',
               'Walnut 1',
               'Watermelon 1',
               'Zucchini 1',
               'Zucchini dark 1']


@app.get("/")
async def root():
    return {"message": "Hello, Server is Running"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


logging.basicConfig(level=logging.DEBUG)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.debug('Received request')
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    json_data = {
        'instances': image_batch.tolist()
    }

    response = requests.post(endpoints, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
