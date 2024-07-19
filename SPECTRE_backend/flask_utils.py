def build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

def build_actual_response(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response