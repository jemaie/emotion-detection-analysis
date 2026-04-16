FROM python:3.12-slim

RUN adduser --disabled-password --no-create-home --shell /sbin/nologin appuser

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Copy app code and static data
COPY .streamlit /app/.streamlit
COPY app.py storage.py export_ratings.py /app/
COPY data /app/data

# Ratings directory (mount as volume for persistence)
RUN mkdir -p /app/ratings && chown appuser:appuser /app/ratings
VOLUME /app/ratings

USER appuser

EXPOSE 8501

# HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
#     CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
