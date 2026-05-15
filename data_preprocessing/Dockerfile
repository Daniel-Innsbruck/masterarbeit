FROM python:3.12-slim

WORKDIR /app

COPY services/guardian_fetcher/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY services/guardian_fetcher/app.py .

CMD ["python", "-u", "app.py"]