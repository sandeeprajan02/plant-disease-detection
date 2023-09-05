from fastapi import FastAPI, UploadFile 
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# model related import statements
import os
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index





UPLOAD_DIR = Path() / 'upload'



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    filename = file_upload.filename
    file_path = os.path.join('upload/', filename)
    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)   
    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description =disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]
    return { "title" : title , "desc" : description, "prevent" : prevent, "sname" : supplement_name, "buy_link" : supplement_buy_link }
































    # data = await file_upload.read()
    # save_to = UPLOAD_DIR / file_upload.filename
    # with open(save_to, 'wb') as f:
    #     f.write(data)
    # return {'filenames':file_upload.filename}
    