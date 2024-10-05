from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__)

# Root route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drawio')
def drawio():
    return render_template('drawio.html')


@app.route('/konva')
def konva():
    return render_template('konva.html')

# Route to save diagram to JSON
@app.route('/save', methods=['POST'])
def save():
    diagram_data = request.json.get('diagram')
    with open('diagram.json', 'w') as f:
        json.dump(diagram_data, f)
    return jsonify({'status': 'success', 'message': 'Diagram saved!'})

# Route to load diagram from JSON
@app.route('/load', methods=['GET'])
def load():
    if os.path.exists('diagram.json'):
        with open('diagram.json', 'r') as f:
            diagram_data = json.load(f)
        return jsonify({'status': 'success', 'diagram': diagram_data})
    else:
        return jsonify({'status': 'error', 'message': 'No saved diagram found!'})

if __name__ == '__main__':
    app.run(debug=True)
