INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city."]
    },
    'duration': {
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [ 5 ],
    },
    'model_type': {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ['medium']
    }
}