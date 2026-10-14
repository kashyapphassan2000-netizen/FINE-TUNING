FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.141.1 "uvicorn[standard]" httpx
COPY app.py .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
