from flask import Flask, request, jsonify
from flask_cors import CORS  # pip install flask-cors
import numpy as np
from main import NounEmbedder, random_word_picker # Import your classes

app = Flask(__name__)
CORS(app) # Allows the browser to talk to this server

# Initialize the embedder once
emb = NounEmbedder.load()
current_target = random_word_picker(emb)
ranks = emb.most_similar(current_target)


@app.route('/get_hint1', methods=['GET'])
def get_hint1():
    rank = emb.most_similar(current_target)
    hint = rank[-1][0]
    return jsonify({"hint": hint, "rank": len(rank)})

@app.route('/get_hint2', methods=['GET'])
def get_hint2():
    rank = emb.most_similar(current_target)
    hint = rank[10][0]
    return jsonify({"hint": hint, "rank": 10})

@app.route('/get_target', methods=['GET'])
def get_target():
    return jsonify({"target": current_target})

@app.route('/get_rank', methods=['GET'])
def get_rank():
    ranks = emb.most_similar(current_target)
    return jsonify({"target": ranks})

@app.route('/set_target', methods=['GET'])
def set_target():
    global current_target
    current_target = random_word_picker(emb)
    return jsonify({"status": "success", "message": "New target set"})

@app.route('/check_word', methods=['POST'])
def check_word():
    word = request.json.get('word', '').lower()
    is_valid = word in emb
    return jsonify({"valid": is_valid})

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    word = request.json.get('word', '').lower()
    if word not in emb:
        return jsonify({"error": "Word not in vocabulary"}), 400
    
    score = emb.similarity(word, current_target)
    # Ensure it's a standard float for JSON serialization
    return jsonify({"score": float(score)})

if __name__ == '__main__':
    app.run(port=5000)