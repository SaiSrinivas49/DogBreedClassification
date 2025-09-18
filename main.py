import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from fastapi import FastAPI,File,UploadFile,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

train_dir = "E:/DogBreedClassification/dogImages/train"
test_dir = "E:/DogBreedClassification/dogImages/test"
valid_dir = "E:/DogBreedClassification/dogImages/valid"

app = FastAPI(title="Dog Breed Classification")

app.mount("/static",StaticFiles(directory="static"),name="static")
templates = Jinja2Templates(directory="templates")

CLASS_NAMES = ['Affenpinscher','Afghan_hound','Airedale_terrier','Akita','Alaskan_malamute','American_eskimo_dog',
               'American_foxhound','American_staffordshire_terrier','American_water_spaniel','Anatolian_shepherd_dog',
               'Australian_cattle_dog','Australian_shepherd','Australian_terrier','Basenji','Basset_hound','Beagle',
               'Bearded_collie','Beauceron','Bedlington_terrier','Belgian_malinois','Belgian_sheepdog','Belgian_tervuren',
               'Bernese_mountain_dog','Bichon_frise','Black_and_tan_coonhound','Black_russian_terrier','Bloodhound',
               'Bluetick_coonhound','Border_collie','Border_terrier','Borzoi','Boston_terrier','Bouvier_des_flandres',
               'Boxer','Boykin_spaniel','Briard','Brittany','Brussels_griffon','Bull_terrier','Bulldog','Bullmastiff',
               'Cairn_terrier','Canaan_dog','Cane_corso','Cardigan_welsh_corgi','Cavalier_king_charles_spaniel',
               'Chesapeake_bay_retriever','Chihuahua','Chinese_crested','Chinese_shar-pei','Chow_chow','Clumber_spaniel',
               'Cock','Field_spaniel','Finnish_spitz','Flat-coated_retriever','French_bulldog','German_pinscher',
               'German_shepherd_dog','German_shorthaired_pointer','German_wirehaired_pointer','Giant_schnauzer',
               'Field_spaniel','Finnish_spitz','Flat-coated_retriever','French_bulldog','German_pinscher',
               'German_shepherd_dog','German_shorthaired_pointer','Field_spaniel','Finnish_spitz','Flat-coated_retriever',
               'French_bulldog','German_pinscher','German_shepherd_dog','German_shorthaired_pointer',
               'German_wirehaired_pointer','Giant_schnauzer','Glen_of_imaal_terrier','Golden_retriever',
               'Gordon_setter','Great_dane','Great_pyrenees','Greater_swiss_mountain_dog','Greyhound','Havanese',
               'Ibizan_hound','Icelandic_sheepdog','Irish_red_and_white_setter','Irish_setter','Irish_terrier',
               'Irish_water_spaniel','Irish_wolfhound','Italian_greyhound','Japanese_chin','Keeshond','Kerry_blue_terrier',
               'Komondor','Kuvasz','Labrador_retriever','Lakeland_terrier','Leonberger','Lhasa_apso','Lowchen','Maltese',
               'Manchester_terrier','Mastiff','Miniature_schnauzer','Neapolitan_mastiff','Newfoundland','Norfolk_terrier',
               'Norwegian_buhund','Norwegian_elkhound','Norwegian_lundehund','Norwich_terrier',
               'Nova_scotia_duck_tolling_retriever','Old_english_sheepdog','Otterhound','Papillon','Parson_russell_terrier',
               'Pekingese','Pembroke_welsh_corgi','Petit_basset_griffon_vendeen','Pharaoh_hound','Plott','Pointer',
               'Pomeranian','Poodle','Portuguese_water_dog','Saint_bernard','Silky_terrier','Smooth_fox_terrier',
               'Tibetan_mastiff','Welsh_springer_spaniel','Wirehaired_pointing_griffon','Xoloitzcuintli','Yorkshire_terrier']

models = {}

@app.on_event("startup")
async def load_models():
    """Load all models into memory on application startup."""
    print("Loading models...")
    models["mobilenetv2"] = tf.keras.models.load_model("dog_breed_classifier_mobilenetv2.h5")
    models["inceptionv3"] = tf.keras.models.load_model("dog_breed_classifier_inceptionv3.h5")
    models["resnet50v2"] = tf.keras.models.load_model("dog_breed_classifier_resnet50v2.h5")
    print("Models loaded successfully!")

# --- Prediction Logic ---
def predict_breed(model_name: str, image_path: str):
    """Preprocesses an image and predicts the dog breed."""
    model = models.get(model_name)
    if not model:
        raise ValueError("Model not found")

    # Model-specific preprocessing
    if model_name == "inceptionv3":
        img_size = (224, 224)
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    elif model_name == "resnet50v2":
        img_size = (224, 224)
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    else: # Default to MobileNetV2
        img_size = (224, 224)
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Return the predicted breed name
    return CLASS_NAMES[predicted_class_index].replace('_', ' ').title()


# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...), model_name: str = Form(...)):
    """Handles file upload, saves it, and shows the confirmation page."""
    # Generate a unique filename to avoid conflicts
    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{file.filename}"
    file_path = os.path.join("static/uploads", filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Render the confirmation page
    return templates.TemplateResponse("confirm.html", {
        "request": request,
        "filename": filename,
        "model_name": model_name
    })


@app.post("/predict", response_class=HTMLResponse)
async def handle_predict(request: Request, filename: str = Form(...), model_name: str = Form(...)):
    """Runs the prediction and shows the final result page."""
    image_path = os.path.join("static/uploads", filename)
    
    # Get the prediction
    predicted_breed = predict_breed(model_name=model_name, image_path=image_path)

    # Render the result page
    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": filename,
        "prediction": predicted_breed
    })
