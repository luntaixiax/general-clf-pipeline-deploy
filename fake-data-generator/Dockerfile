FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && apt-get autoremove

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .