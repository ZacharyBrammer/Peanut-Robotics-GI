FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git && apt-get install -y wget

RUN wget https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights -P ./Utils
 
COPY ./requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR app/

COPY . /app
 
CMD ["streamlit", "run", "Display.py", "--server.port", "8501"]