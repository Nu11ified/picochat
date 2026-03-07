FROM rust:latest AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/picochat-cli /app/picochat
COPY --from=builder /app/web /app/web

RUN mkdir -p /app/model && \
    curl -L "https://huggingface.co/manasred/picochat/resolve/main/model.safetensors" -o /app/model/model.safetensors && \
    curl -L "https://huggingface.co/manasred/picochat/resolve/main/config.json" -o /app/model/config.json && \
    curl -L "https://huggingface.co/manasred/picochat/resolve/main/tokenizer.json" -o /app/model/tokenizer.json

EXPOSE 8000

CMD ["/app/picochat", "--serve", "--load", "/app/model", "--tokenizer", "/app/model/tokenizer.json", "--port", "8000"]
