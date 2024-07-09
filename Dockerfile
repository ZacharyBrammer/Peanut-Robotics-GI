FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git
 
COPY ./requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
WORKDIR app/
 
COPY . /app
# COPY ./.streamlit /app/.streamlit
 
CMD ["streamlit", "run", "Display.py", "--server.port", "8080"]

# # FROM python:3.12
# FROM python:3.9

# # ensure local python is preferred over distribution python
# # ENV PATH /usr/local/bin:$PATH
# WORKDIR /

# RUN pip install --upgrade pip

# COPY . .

# RUN pip install -r requirements.txt
# # RUN pip install pillow datasets lancedb
# # RUN pip install git+https://github.com/openai/CLIP.git
# # RUN pip install numpy
# # RUN pip install tqdm
# # RUN pip install torch
# # RUN pip install pyarrow
# # RUN pip install datasets
# # RUN pip install ipython
# # RUN pip install pandas
# # RUN pip install pillow
# # RUN pip install requests
# # RUN pip install regex
# # RUN pip install altair
# # RUN pip install streamlit
# # RUN pip install matplotlib

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# # ENTRYPOINT ["streamlit", "run", "Display.py"]
# ENTRYPOINT ["streamlit", "run", "Display.py", "--server.port=8501", "--server.address=0.0.0.0"]

# # CMD streamlit run Display.py