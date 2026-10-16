# Real-Time ML Inference REST API & Capstone

A container-ready FastAPI service that exposes a sentiment-analysis model through a JSON REST API. It includes model artifact creation/loading, prediction probabilities, schema validation, automated tests, and Docker configuration.

## Architecture

```text
Client (JSON)
    |
    v
FastAPI /predict
    |
    v
Input validation (Pydantic)
    |
    v
TF-IDF vectorizer
    |
    v
Trained MLPClassifier
    |
    v
Label + confidence + class probabilities (JSON)
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic service information |
| GET | `/health` | Health and model-loaded status |
| POST | `/predict` | Classify one or more text samples |

Interactive OpenAPI documentation is available at `/docs` while the service is running.

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this product","This is terrible"]}'
```

Example response structure:

```json
{
  "predictions": [
    {
      "text": "I love this product",
      "label": "Positive",
      "confidence": 0.81,
      "probabilities": {
        "Negative": 0.19,
        "Positive": 0.81
      }
    }
  ],
  "model_version": "1.0.0"
}
```

The exact confidence values depend on the trained model and library version.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API.

The first startup trains a small demonstration model and writes local pickle artifacts. Those artifacts are intentionally excluded from Git; they are created automatically when the service starts. The bundled dataset is illustrative only and is not suitable for production decisions.

## Run tests

```bash
pytest -q
```

Tests cover health checks, the root endpoint, prediction response schema, probability ranges/sums, and invalid inputs.

## Run with Docker

```bash
docker build -t sentiment-inference-api .
docker run --rm -p 8000:8000 sentiment-inference-api
```

Or use Docker Compose:

```bash
docker compose up --build
```

Then visit `http://localhost:8000/docs`.

## Request validation and error handling

- `texts` is required and must contain between 1 and 100 strings.
- Blank or whitespace-only text is rejected with HTTP 422.
- Successful predictions return a label, top-class confidence, and probabilities for both classes.
- FastAPI automatically publishes an OpenAPI schema and interactive Swagger UI.

## Model and limitations

The API uses scikit-learn's `MLPClassifier` with TF-IDF features as a portable demonstration model. Replace the sample training data with a properly sourced, larger dataset and evaluate generalization before production use. Pickle files should only be loaded from trusted sources. For real deployment, add authentication, rate limiting, structured logging, monitoring, and model/data versioning.
