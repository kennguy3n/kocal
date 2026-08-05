FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o kchat-server-offload .

FROM alpine:3.19
RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /app/kchat-server-offload .

EXPOSE 8080
ENV PORT=8080
ENV GIN_MODE=release

CMD ["./kchat-server-offload"]
