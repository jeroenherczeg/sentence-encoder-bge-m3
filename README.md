# Sentence Encoder bge-m3

FastAPI implementation of BAAI/bge-m3 encoder, containerized for scalable Kubernetes deployment.

## Description

This project provides a high-performance API for generating sentence embeddings using the BAAI/bge-m3 model. It's built with FastAPI for efficient handling of requests and containerized for easy deployment and scaling in Kubernetes environments.

## Features

- Fast and efficient sentence encoding using BAAI/bge-m3 model
- RESTful API built with FastAPI
- Docker containerization for consistent environments
- Kubernetes deployment ready
- Scalable architecture suitable for high-load environments
- Health check endpoints for Kubernetes probes

## Prerequisites

- Python 3.9+
- Docker
- Kubernetes cluster (for production deployment)

## Quick Start

### Local Development

1. Clone the repository:
   ```
   git clone git@github.com:jeroenherczeg/sentence-encoder-bge-m3.git
   cd sentence-encoder-bge-m3
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

4. Access the API at `http://localhost:8000` and the interactive docs at `http://localhost:8000/docs`

### Docker

1. Build the Docker image:
   ```
   docker build -t sentence-encoder-bge-m3:latest .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 sentence-encoder-bge-m3:latest
   ```

### Kubernetes Deployment

1. Apply the Kubernetes manifests:
   ```
   kubectl apply -f kubernetes/
   ```

2. Access the service (method depends on your Kubernetes setup)

## API Usage

### Encode Sentences

**Endpoint**: `POST /encode`

```bash
curl -X POST "http://localhost:8000/encode" -H "Content-Type: application/json" -d '{"sentences": ["Hello, world!", "This is a test sentence."]}'
```

**Request Body**:
```json
{
  "sentences": ["Hello, world!", "Another sentence to encode."]
}
```

**Response**:
```json
{
  "encodings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

### Kubernetes Health Probes

**Endpoint**: `GET /readiness`

```bash
curl http://localhost:8000/readiness
```

**Endpoint**: `GET /liveness`

```bash
curl http://localhost:8000/liveness
```

## Model Information

This API uses the BAAI/bge-m3 model, which is a state-of-the-art sentence embedding model. It's designed to generate high-quality vector representations of sentences that capture semantic meaning, making it ideal for various natural language processing tasks such as semantic search, text classification, and similarity comparison.

## Performance

The BAAI/bge-m3 model offers a good balance between performance and accuracy. In our testing, it processes approximately 457 sentences per second on a standard CPU. For production environments, we recommend GPU acceleration for higher throughput.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for the bge-m3 model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for the embedding framework
- [Docker](https://www.docker.com/) for containerization
- [Kubernetes](https://kubernetes.io/) for orchestration