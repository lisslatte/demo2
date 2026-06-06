FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV MODEL_DIR=/app/fly/flight_delay_model_artifacts
ENV AIRPORTS_PATH=/app/airports.csv

COPY flight_delay_business_recommendation_demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY flight_delay_business_recommendation_demo /app/flight_delay_business_recommendation_demo
COPY fly/flight_delay_pipeline.py /app/fly/flight_delay_pipeline.py
COPY fly/flight_delay_model_artifacts /app/fly/flight_delay_model_artifacts
COPY airports.csv /app/airports.csv

EXPOSE 8501

CMD ["sh", "-c", "streamlit run flight_delay_business_recommendation_demo/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
