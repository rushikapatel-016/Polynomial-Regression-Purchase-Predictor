from pathlib import Path

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Fashion_Retail_Sales.csv"

app = Flask(__name__)

# Load dataset
data = pd.read_csv(DATA_PATH)

data["Review Rating"] = pd.to_numeric(
    data["Review Rating"], errors="coerce"
)
data["Purchase Amount (USD)"] = pd.to_numeric(
    data["Purchase Amount (USD)"], errors="coerce"
)

data = data.dropna(
    subset=["Review Rating", "Purchase Amount (USD)"]
)

# Keep valid ratings
data = data[data["Review Rating"].between(0, 5)]

X = data[["Review Rating"]]
y = data["Purchase Amount (USD)"]

# Polynomial Regression model
model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2)),
    ("regression", LinearRegression())
])

model.fit(X, y)

# Create smooth prediction curve
ratings = pd.DataFrame({
    "Review Rating": [i / 20 for i in range(101)]
})
curve = model.predict(ratings)

# Actual dataset points
actual = data.sample(
    n=min(300, len(data)), random_state=42
).sort_values("Review Rating")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Polynomial Regression Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial;
            background: #eef2f7;
            text-align: center;
            padding: 30px 15px;
        }
        .box {
            background: white;
            max-width: 750px;
            margin: auto;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px #cccccc;
        }
        input, button {
            width: 90%;
            padding: 12px;
            margin-top: 15px;
            box-sizing: border-box;
        }
        button {
            background: #3157b7;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            color: #176534;
            font-weight: bold;
        }
        .graph {
            margin-top: 35px;
        }
    </style>
</head>
<body>
<div class="box">
    <h2>Purchase Amount Predictor</h2>
    <p>Polynomial Regression Machine Learning Project</p>

    <form method="POST">
        <label>Enter Review Rating (0 to 5)</label><br>
        <input type="number" name="rating"
               min="0" max="5" step="0.1"
               value="{{ rating }}" required>
        <button type="submit">Predict Amount</button>
    </form>

    {% if prediction is not none %}
        <p class="result">
            Estimated Purchase Amount: ${{ prediction }}
        </p>
    {% endif %}

    {% if error %}
        <p style="color:red">{{ error }}</p>
    {% endif %}

    <div class="graph">
        <h3>Polynomial Regression Graph</h3>
        <canvas id="myChart"></canvas>
    </div>
</div>

<script>
const actualData = {{ actual_data | tojson }};
const curveData = {{ curve_data | tojson }};

new Chart(document.getElementById("myChart"), {
    type: "scatter",
    data: {
        datasets: [
            {
                label: "Actual Data",
                data: actualData,
                backgroundColor: "blue",
                pointRadius: 3
            },
            {
                label: "Polynomial Regression Curve",
                data: curveData,
                type: "line",
                borderColor: "red",
                borderWidth: 3,
                pointRadius: 0,
                showLine: true,
                tension: 0
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                type: "linear",
                title: {
                    display: true,
                    text: "Review Rating"
                },
                min: 0,
                max: 5
            },
            y: {
                title: {
                    display: true,
                    text: "Purchase Amount (USD)"
                }
            }
        }
    }
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    rating = ""
    error = ""

    if request.method == "POST":
        try:
            rating = float(request.form.get("rating", ""))

            if not 0 <= rating <= 5:
                error = "Rating must be between 0 and 5."
            else:
                result = model.predict(
                    pd.DataFrame({"Review Rating": [rating]})
                )[0]
                prediction = round(float(result), 2)

        except ValueError:
            error = "Please enter a valid rating."

    actual_data = [
        {"x": float(row["Review Rating"]),
         "y": float(row["Purchase Amount (USD)"])}
        for _, row in actual.iterrows()
    ]

    curve_data = [
        {"x": float(x), "y": float(y)}
        for x, y in zip(ratings["Review Rating"], curve)
    ]

    return render_template_string(
        HTML,
        prediction=prediction,
        rating=rating,
        error=error,
        actual_data=actual_data,
        curve_data=curve_data
    )


if __name__ == "__main__":
    print("Polynomial Regression model trained!")
    print("Valid records:", len(data))
    app.run(debug=True)
