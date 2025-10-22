from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_KEY')
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    return jsonify({"answer": f"You asked: {question}"})

if __name__ == "__main__":
    app.run(debug=True)
