from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/save', methods=['POST'])
def save_grid():
    data = request.json
    filename = data.get("fname", "")
    if filename == ".json":
        return jsonify({"success": False, "error": "Invalid file name"})
    with open("dataset/"+filename, 'w') as f:
        grid_entries = data.get("grid", [])
        content = [
            {k: v for k, v in entry.items()}
            for entry in grid_entries
        ]
        json.dump({
            "layers": content,
            "dim": data.get("dim")
        }, f)
    return jsonify({'success': True, 'file': filename})

@app.route('/load', methods=['POST'])
def load_grid():
    data = request.json
    filename = "dataset/"+data.get("fname", "")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            grid_data = json.load(f)
        rows, cols = tuple(grid_data["dim"])
        return jsonify({'success':True, 'layers': grid_data["layers"], "rows":rows, "cols":cols})
    return jsonify({'success':False, 'error': 'File not found'})

if __name__ == '__main__':
    app.run(port=5002, debug=True)