#FROM rayproject/ray:2.5.0.142b46-py39-cu116
#FROM rayproject/ray:2.7.1.d7e0ac-py39-cu117
#FROM rayproject/ray:2.7.1.artur.3c57d9-py39-cu117
FROM rayproject/ray-ml:2.20.0.5708e7-py39-cu118

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]