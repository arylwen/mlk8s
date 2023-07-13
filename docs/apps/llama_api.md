# llama api

llama api is an openai compatible web service. It uses RAY to enable multi-cpu inference. below are the instructions on how to run locally, build and push the docker image.

## create conda env

pyvis is not compatible with python 3.10 and above.
```
conda create -n ray39 python=3.9.15 -y 
conda actvate ray39 
```

## apply dev requirements
```
pip install -r requirements-dev.txt 
```
