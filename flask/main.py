from flask import Flask, request, jsonify # Se utiliza para crear y gestionar la aplicación web, Convierte objetos Python en respuestas JSON.
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo previamente guardado
#with open('modelo_clasificacion.pkl', 'rb') as f:
   # model = pickle.load(f)

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo
with open('transformers/mejor_modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar los transformers
with open('transformers/transformer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('transformers/transformer_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open('transformers/transformer_one.pkl', 'rb') as f:
    one = pickle.load(f)

variables_one = ["Gender", "ProductCategory"]

# Definir el endpoint de la raíz (/) para comprobar que el servidor funciona. 
@app.route('/')
def home():
    return jsonify({
        "message": "API de predicción en funcionamiento",
        "endpoints": {
            "/predict": "Usa esta ruta para realizar predicciones (método POST)"
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Recibe datos en formato JSON
        if not data:
            return jsonify({"error": "No features provided"}), 400
        df_pred = pd.DataFrame(data, index = [0])
        df_pred["DiscountsAvailed"] = df_pred["DiscountsAvailed"].astype("category")

        # saco las columnas numericas y estandarizo
        col_numericas = df_pred.select_dtypes(include = np.number).columns

        df_pred[col_numericas] = scaler.transform(df_pred[col_numericas])

        # transformamos las categorias one-hot
        df_one = pd.DataFrame(one.transform(df_pred[variables_one]).toarray(), columns=one.get_feature_names_out())
        df_pred = pd.concat([df_pred, df_one], axis = 1)
        df_pred.drop(columns = variables_one, axis = 1, inplace = True)

        # transformamos las categoricas de target
        df_pred = target.transform(df_pred)

        prediction = model.predict(df_pred)
        prob = model.predict_proba(df_pred)
        print(prob)
        return jsonify({
            "prediction": prediction.tolist()[0],
            "probabilities": round(prob.tolist()[0][1],2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)