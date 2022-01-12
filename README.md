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

- URL
  `/eval`
- Method
  `GET`
- Params
  `NONE`
- Response
  - Code: `200`
  - Content: `{ img : *image encoded* }`

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
