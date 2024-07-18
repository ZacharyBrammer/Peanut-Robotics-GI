FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8501
 
EXPOSE 8501
 
RUN apt-get update && apt-get install -y git && apt-get install -y wget

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR app/

COPY . /app

ADD https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights?download= /app/Utils

CMD ["streamlit", "run", "Display.py", "--server.port", "8501"]