from flask import Flask, request, jsonify
import model
from model import model
import rl_agent
from rl_agent import dynamic_pricing, static_pricing


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # set max content length to 100MB

@app.route('/')
def home():
    return "hello"

@app.route('/predict_time_series', methods=['POST'])
def predict_time_series():
    data = request.json['data']
    try:
        res = model(data)
        return jsonify({'result': res}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_dynamic_pricing', methods=['POST'])
def predict_dynamic_pricing():
    data = request.json['data']
    try:
        print("data:",data)
        result_actions, result_rewards, total, result_daily_rewards, result_daily_actions =  dynamic_pricing(data)
        # create a dictionary containing the values to be returned
        response = {
            'result': {
                'result_actions': result_actions,
                'result_rewards': result_rewards,
                'total_dynamic_revenue': total
            }
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_static_pricing', methods=['POST'])
def predict_static_pricing():
    data = request.json['data']
    try:
        print("data:",data)
        result_rewards_static, total_static, set_price = static_pricing(data)
        # create a dictionary containing the values to be returned
        response = {
            'result': {
                'result_rewards_static': result_rewards_static,
                'total_static_revenue': total_static
            }
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
