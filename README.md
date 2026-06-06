# Flight Delay Passenger Demo

Public deployment package for the Streamlit passenger flight-delay demo.

## Deploy On Render

1. Create a new Render Web Service from this GitHub repository.
2. Choose Docker runtime.
3. Set the Dockerfile path:

```text
flight_delay_business_recommendation_demo/Dockerfile
```

4. Add this environment variable in Render:

```text
AVIATIONSTACK_API_KEY=your_key_here
```

5. Deploy.

## Run Locally With Docker

```powershell
docker build -f flight_delay_business_recommendation_demo/Dockerfile -t flight-delay-demo .
docker run --rm -p 8501:8501 -e AVIATIONSTACK_API_KEY="your_key_here" flight-delay-demo
```

Open:

```text
http://localhost:8501
```

## Included Files

- Streamlit app in `flight_delay_business_recommendation_demo/`
- ML model artifacts in `fly/flight_delay_model_artifacts/`
- Airport lookup data in `airports.csv`
- Docker deployment setup

Do not commit API keys into this repository. Add them only as environment variables or Streamlit secrets.
