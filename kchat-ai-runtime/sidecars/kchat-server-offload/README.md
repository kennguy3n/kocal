# KChat Server-Side Offload Service

A Go service that handles AI inference requests when the on-device runtime
cannot process them (low-tier device, thermal throttling, battery saver,
or model not installed).

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  KChat App  │────▶│  Server Offload  │────▶│  llama.cpp /    │
│  (on-device)│     │  (Go service)    │     │  Cloud LLM API  │
└─────────────┘     └──────────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Vector DB   │
                    │  (Qdrant)    │
                    └──────────────┘
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/safety/classify` | Classify text for safety |
| POST | `/api/v1/context/retrieve` | Retrieve context documents |
| POST | `/api/v1/generation/generate` | Generate text with grammar constraints |
| POST | `/api/v1/action/validate` | Validate a tool plan |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/models` | List available models |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `PORT` | `8080` | Server port |
| `KCHAT_API_KEY` | (empty) | Bearer token for authentication |
| `GIN_MODE` | `release` | Gin mode (release/debug) |
| `LLAMA_SERVER_URL` | `http://localhost:18888` | llama.cpp server URL |
| `VECTOR_DB_URL` | `http://localhost:6333` | Qdrant vector DB URL |

## Running

```bash
# Set API key
export KCHAT_API_KEY=your-secret-key

# Run the service
go run .

# Or build and run
go build -o kchat-server-offload
./kchat-server-offload
```

## Docker

```bash
docker build -t kchat-server-offload .
docker run -p 8080:8080 -e KCHAT_API_KEY=secret kchat-server-offload
```

## Example Request

```bash
# Classify safety
curl -X POST http://localhost:8080/api/v1/safety/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","is_group":false}'

# Generate text
curl -X POST http://localhost:8080/api/v1/generation/generate \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize this: ...","max_tokens":128}'
```
