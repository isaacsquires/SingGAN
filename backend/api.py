from flask import Flask
from flask_restful import Resource, Api, reqparse
from model.evaluate import evaluate
from model.train import train


app = Flask(__name__)
api = Api(app)


class EvaluateModel(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('audio', type=list, location="json")
        audio = parse.parse_args()
        img = evaluate(audio)
        return {'img': img}


class TrainModel(Resource):
    def get(self):
        raw = train()
        return 'Training complete'


api.add_resource(EvaluateModel, '/eval')
api.add_resource(TrainModel, '/train')

if __name__ == '__main__':
    app.run(debug=True)
