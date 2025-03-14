
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = "roadaccidentai.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Serve the HTML page
@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Road Accident Prediction</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: rgb(53, 98, 117); 
                text-align: center; 
                margin: 50px; 
            }
            form { 
                width : 900px;
                display: inline-block; 
                font-family: 'Poppins', sans-serif;
                text-align: left; 
                padding: 20px;
                border: 1px solid #ccc; 
                border-radius: 10px; 
                background:  #fff;
                color: #333;
             }
            select, button { 
                padding: 10px; 
                margin: 5px; 
                width: 100%; 
            }
            h3 { 
                margin-top: 20px; 
                background: rgb(53, 98, 117);
            }
        </style>
    </head>
    <body>
        <h1>Road Accident Prediction</h1>
        <form id="prediction-form">
            <label>Age Band of Driver</label>
            <select id="Age_band_of_driver">
                <option value="18">Under 18</option>
                <option value="31">18-30</option>
                <option value="0">31-50</option>
                <option value="50">Over 51</option>
            </select>

            <label>Sex of Driver</label>
            <select id="Sex_of_driver">
                <option value="1">Female</option>
                <option value="0">Male</option>
                <option value="2">others</option>
            </select>

            <label>Educational Level</label>
            <select id="Educational_level">
                <option value="1">Above high school</option>
                <option value="2">Junior high school</option>
                <option value="3">Writing & reading</option>
                <option value="4">Elementary school</option>
                <option value="5">High school</option>
                <option value="6">Unknown</option>
            </select>

            <label>Vehicle Driver Relation</label>
            <select id="Vehicle_driver_relation">
                <option value="0">Employee</option>
                <option value="2">Unknown</option>
                <option value="1">Owner</option>
                <option value="4">Other</option>
            </select>

            <label>Driving Experience</label>
            <select id="Driving_experience">
                <option value="1">1-2yr</option>
                <option value="2">Above 10yr</option>
                <option value="3">5-10yr</option>
                <option value="4">2-5yr</option>
                <option value="5">Unknown</option>
                <option value="6">No License</option>
                <option value="7">Below 1yr</option>
            </select>

            <label>Lanes or Medians</label>
            <select id="Lanes_or_Medians">
                <option value="1">Undivided Two way</option>
                <option value="2">Unknown</option>
                <option value="3">Double carriageway(median)</option>
                <option value="4">Two_way(divided with solid lines road marking)</option>
            </select>

            <label>Types of Junction</label>
            <select id="Types_of_Junction">
                <option value="0">No junction</option>
                <option value="2">Y Shape</option>
                <option value="1">Crossing</option>
                <option value="4">O Shape</option>
                <option value="3">Other</option>
                <option value="6">Unknown</option>
            </select>

            <label>Road Surface Type</label>
            <select id="Road_surface_type">
                <option value="1">Asphalt roads</option>
                <option value="2">Earth roads</option>
                <option value="3">Gravel roads</option>
                <option value="4">Unknown</option>
                <option value="5">Other</option>
            </select>

            <label>Light Conditions</label>
            <select id="Light_conditions">
                <option value="0">Daylight</option>
                <option value="1">Darkness - lights lit</option>
            </select>

            <label>Weather Conditions</label>
            <select id="Weather_conditions">
                <option value="0">Normal</option>
                <option value="1">Raining</option>
                <option value="4">Raining and Windy</option>
                <option value="3">Other</option>
                <option value="5">Cloudy</option>
                <option value="6">Windy</option>
                <option value="7">Snow</option>
                <option value="2">Fog or mist</option>
            </select>

            <button type="submit", style="font-size : 20px; font-weight: bold; font-family: 'Poppins', sans-serif;">Predict</button>
        </form>

        <h3 id="prediction-result"></h3>

        <script>
            document.getElementById("prediction-form").addEventListener("submit", function(event) {
                event.preventDefault();

                let formData = {
                    Age_band_of_driver: document.getElementById("Age_band_of_driver").value,
                    Sex_of_driver: document.getElementById("Sex_of_driver").value,
                    Educational_level: document.getElementById("Educational_level").value,
                    Vehicle_driver_relation: document.getElementById("Vehicle_driver_relation").value,
                    Driving_experience: document.getElementById("Driving_experience").value,
                    Lanes_or_Medians: document.getElementById("Lanes_or_Medians").value,
                    Types_of_Junction: document.getElementById("Types_of_Junction").value,
                    Road_surface_type: document.getElementById("Road_surface_type").value,
                    Light_conditions: document.getElementById("Light_conditions").value,
                    Weather_conditions: document.getElementById("Weather_conditions").value
                };

                fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(formData)
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById("prediction-result").innerText = 
                        data.result ? `Accident Probability: ${data.result}%` : "Error: " + data.error;
                })
                .catch(error => console.error("Error:", error));
            });
        </script>
    </body>
    </html>
    """

# API route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not found. Please check the file path."})

        data = request.get_json()
        features = [
            int(data["Age_band_of_driver"]),
            int(data["Sex_of_driver"]),
            int(data["Educational_level"]),
            int(data["Vehicle_driver_relation"]),
            int(data["Driving_experience"]),
            int(data["Lanes_or_Medians"]),
            int(data["Types_of_Junction"]),
            int(data["Road_surface_type"]),
            int(data["Light_conditions"]),
            int(data["Weather_conditions"]),
        ]

        other_features = [0] * 4
        features = np.array(features + other_features).reshape(1, -1)

        probability = model.predict(features)  # Get prediction (numpy array)
        print(probability)
        percentage = (probability.item() / 2) * 100  # Convert severity (0-2) to percentage (0-100)
        probability_percentage = round(percentage, 2)  # Round to 2 decimal places

        print(probability_percentage)  # Output the percentage

        return jsonify({"result": probability_percentage})  # Convert to percentage

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

