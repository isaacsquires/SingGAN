# SingGAN

## Live audio to image generative adverserial network

## Backend

Running Python backend first time

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run all Python scripts from `./backend` in virtual env.

### Flask API

To run Flask API

```bash
python api.py
```

#### Endpoints

##### EvaluateModel

Evaluate the model using the saved params and return output of the generator as base64 encoded image.

- URL
  `/eval`
- Method
  `POST`
- Params
  `audio: <array length 256>`
- Response
  - Code: `200`
  - Content: `{ img : *image encoded* }`

##### TrainModel

Initiate training of model. Saves trained model to `model/saved_models` directory, creates directory if not created.
TODO - actual training rather than dummy training

- URL
  `/train`
- Method
  `GET`
- Params
  `NONE`
- Response
  - Code: `200`
  - Content: `{modelTrained: True}`

## Structure

```
.
├── README.md
├── backend
│   ├── model
│   │   ├── config.py
│   │   ├── network.py
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   └── saved_models
│   ├── requirements.txt
│   ├── .gitignore
│   └── api.py
├── frontend
    ├── ...

```
