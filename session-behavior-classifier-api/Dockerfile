FROM python:3.6-slim
COPY . /pythonModelAPI
WORKDIR /pythonModelAPI
RUN pip install -r requirements.txt
ENTRYPOINT ["flask","run"]