FROM python:3.8.10-slim

WORKDIR /usr/src/app

COPY main.py /usr/src/app/
COPY model.py /usr/src/app/
COPY requirements.txt /usr/src/app/
COPY weights/ /usr/src/app/weights/
RUN ls -la /usr/src/app/weights/*


RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501"]