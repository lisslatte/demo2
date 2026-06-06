# Flight Delay Business Recommendation Demo

Interactive Streamlit prototype that converts a flight delay probability into practical business recommendations for airports, passengers, and airlines.

## Run

```powershell
cd D:\FYP\flight_delay_business_recommendation_demo
streamlit run app.py
```

## Deploy

The demo can be containerized from the project root:

```powershell
cd D:\FYP
docker build -f flight_delay_business_recommendation_demo/Dockerfile -t flight-delay-business-demo .
docker run --rm -p 8501:8501 flight-delay-business-demo
```

Open <http://localhost:8501> after the container starts.

The app also supports configurable artifact paths for hosted deployments:

```text
MODEL_DIR=/path/to/flight_delay_model_artifacts
MODEL_PATH=/path/to/best_delay_model_pipeline.joblib
METADATA_PATH=/path/to/best_delay_model_metadata.json
THRESHOLDS_PATH=/path/to/best_model_thresholds.csv
AIRPORTS_PATH=/path/to/airports.csv
AVIATIONSTACK_API_KEY=optional_key_for_flight_lookup
```

## What This Version Adds

- Delay probability and delay classification.
- User-friendly flight entry by route/date/time, plus optional Aviationstack flight lookup when `AVIATIONSTACK_API_KEY` is configured.
- Automatic origin-airport weather integration using the no-key Open-Meteo API, with weather details shown to users.
- Automatic airport-load and recent-delay pressure estimates from route, time, weather severity, and holiday proximity.
- Three fixed threshold modes: Recall first, Balanced, and Precision first.
- Airport recommendations for gate, ramp, staffing, network route monitoring, and passenger-flow planning.
- Passenger segmentation into Critical Passengers, At-Risk Standard Passengers, and Flexible Passengers.
- Airline recommendations for operational review, crew planning, aircraft recovery, connection protection, and early rebooking communication.
- Batch CSV scoring with business action columns.

## Model

The demo uses the saved model artifact from:

```text
..\fly\flight_delay_model_artifacts\best_delay_model_pipeline.joblib
```
