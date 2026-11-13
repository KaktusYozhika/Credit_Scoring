# Учебный сервис кредитного скоринга: 2 версии для демонстрации отката.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY artifacts/versions/ ./artifacts/versions/
ARG MODEL_VERSION=v1.0.0
ENV MODEL_VERSION=${MODEL_VERSION}
RUN test -f "artifacts/versions/${MODEL_VERSION}/model.joblib"
EXPOSE 8077
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8077/health'))['status'] == 'ok'"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
