import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import make_column_transformer
from joblib import load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

app = FastAPI()

#random gesorteerde versie van de mushroom dataset
mushroom = pd.read_csv('mushroom_random_4columns.csv', sep=',')

# Scheid de doelvariabele 'class' van de kenmerken in de 'mushroom' dataset
X = mushroom.drop('class', axis=1)
y = mushroom['class']

# Verdeel de gegevens in trainings- en testsets met een verhouding van 80% trainen en 20% testen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Laad het getrainde Random Forest Classifier model
model = load('model1_rfc_4columns.joblib')

# Definieer het invoerdatamodel
class MushroomInput(BaseModel):
    stem_width: float
    cap_diameter: float
    cap_shape: str
    cap_color: str

# Maak een kolomtransformator aan voor de 4 belangrijkste kenmerken
important_columns = ['cap-diameter', 'cap-shape', 'cap-color', 'stem-width']
important_categorical_columns = ['cap-shape', 'cap-color']
important_numeric_columns = ['cap-diameter', 'stem-width']

important_column_transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), important_categorical_columns),
    remainder='passthrough'
)

# Fit de kolomtransformator op de trainingsdata
important_column_transformer.fit(X_train)

# Encode de target variable via LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

@app.post("/predict")
def predict_mushroom(data: MushroomInput):
    # Maak een DataFrame met de 4 belangrijkste kenmerken
    input_data = pd.DataFrame([{
        'cap-diameter': data.cap_diameter,
        'cap-shape': data.cap_shape,
        'cap-color': data.cap_color,
        'stem-width': data.stem_width
    }], columns=important_columns)

    # Transformeer invoergegevens met behulp van de belangrijkste kolomtransformator
    input_data_transformed = important_column_transformer.transform(input_data)

    # Maak de voorspelling
    prediction = model.predict(input_data_transformed)

    # Bereken het vertrouwen percentage
    confidence = model.predict_proba(input_data_transformed)
    confidence_percentage = np.max(confidence) * 100

    # Decodeer de voorspelling
    prediction_decoded = label_encoder.inverse_transform(prediction)

    if(prediction_decoded[0] == 'p'):
        return {"message": "Deze paddenstoel is giftig", "confidence": f"{confidence_percentage:.2f}%"}
    else:
        return {"message": "Deze paddenstoel is eetbaar", "confidence": f"{confidence_percentage:.2f}%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)