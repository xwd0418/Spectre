from flask import Flask, jsonify
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS so that the frontend can access the backend
CORS(app)

# Dummy route to test connection
@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"message": "Hello from the backend!"}), 200

if __name__ == '__main__':
    # Run the Flask app on localhost at port 6660
    from waitress import serve
    # app.run(host='0.0.0.0', port=6660)
    serve(app, host="0.0.0.0", port=6660)
