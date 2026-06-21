FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY utils/ ./utils/
COPY training/ ./training/
COPY settings.py .
COPY mlruns/ ./mlruns/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
