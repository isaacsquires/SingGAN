from flask import Flask
from flask_restful import Resource, Api, reqparse
from model.evaluate import evaluate
from model.train import train
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('x')


class EvaluateModel(Resource):
    def get(self):
        args = parser.parse_args()
        x = args['x']
        img_url = 'static/img.png'
        img = evaluate(img_url)

        return {'img': img}


class TrainModel(Resource):
    def get(self):
        args = parser.parse_args()
        x = args['x']
        raw = train()
        return {'raw': raw}


api.add_resource(EvaluateModel, '/eval')
api.add_resource(TrainModel, '/train')

if __name__ == '__main__':
    app.run(debug=True)
