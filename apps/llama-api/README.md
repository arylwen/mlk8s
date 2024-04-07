# llama api

llama api is an openai compatible web service. It uses RAY to enable multi-cpu inference. Below are the instructions on how to run locally, build and push the docker image.

## create conda env

pyvis is not compatible with python 3.10 and above.

```bash
conda create -n ray271 python=3.9 -y 
conda activate ray271 
```

## apply dev requirements

```bash
cd mlk8s/apps/llama-api 
pip install -r requirements-dev.txt 
```

## run locally

Start a new console tab.

```bash
cd mlk8s/apps/llama-api 
conda activate ray271

set-title LA-RUN 
uvicorn app.main:app --reload 
```

Access the service at:
[http://localhost:8000/docs](http://localhost:8000/docs)

## build docker image

```bash
set-title LA-BUILD 
docker build -t registry.local:32000/llama-api:latest . 
```

## push image

```bash
set-title LA-PUSH 
docker push registry.local:32000/llama-api:latest 
```
