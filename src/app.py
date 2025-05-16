from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Bank Nifty Trading System"

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "bank-nifty-trading-system"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
