from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

# Load model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Hate Speech Detection</title>
</head>
<body>
    <h2>Hate Speech Detection</h2>
    <form method="POST">
        <textarea name="text" rows="5" cols="50"></textarea><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>{{ prediction }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        result = model.predict([text])[0]

        if result == 1:
            prediction = " Hate Speech Detected"
        else:
            prediction = " Normal Speech"

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)