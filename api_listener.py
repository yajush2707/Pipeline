from flask import Flask, jsonify, request, abort
import base64
import string
import random

# further description of flask method can be found here: 
# https://stackoverflow.com/questions/10434599/get-the-data-received-in-a-flask-request

# request

api = Flask(__name__)

@api.route('/api', methods=['GET'])
def get_data():
    data = request.values.to_dict(flat=False)
    result = jsonify({'msg': 'success', 'data': str(data)})
    return result
@api.route('/api', methods=['POST'])
def create_task():
    data = request.values.to_dict(flat=False)
    data_json = request.json
    result = jsonify({'msg': 'success', 'data': data,'json': data_json})
    return result


if __name__ == '__main__':
    api.run(host= '0.0.0.0')
