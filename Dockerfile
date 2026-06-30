FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py .
COPY models/model.pkl ./models/model.pkl
COPY models/metrics.json ./models/metrics.json

ENV MODEL_PATH=./models/model.pkl
ENV METRICS_PATH=./models/metrics.json

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
