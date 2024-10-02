from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/ai/layout', methods=['GET'])
def create_layout():
    # ai logic goes here

    data = {
        'message': 'Hello, mazafaqa!',
        'status': 'success'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)