FROM python:3.11-slim
WORKDIR /worker
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker ./worker
CMD ["python", "worker/camera_consumer.py"]
