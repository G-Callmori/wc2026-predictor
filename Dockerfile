FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY WorldCupPredictor2026.py .
COPY wc2026_api.py .

EXPOSE 7860

CMD ["uvicorn", "wc2026_api:app", "--host", "0.0.0.0", "--port", "7860"]
