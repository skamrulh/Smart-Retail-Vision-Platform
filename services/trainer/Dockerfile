FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /train
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train ./train
CMD ["python", "train/train.py"]
