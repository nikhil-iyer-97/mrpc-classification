from flask import Flask
from flask import request

import test
from test import test_example

app = Flask(__name__)

@app.route('/', methods=['POST'])

def index():
    sentence1 = request.get_json()['input1']
    sentence2 = request.get_json()['input2']
    result = test_example(sentence1, sentence2)
    print(result)
    print("")
    return result


if __name__ == '__main__':
   app.run(debug = True, host = '0.0.0.0')
