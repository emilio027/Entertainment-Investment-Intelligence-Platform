# Entertainment Investment Intelligence Platform
## Deployment Guide

### Version 2.0.0 Enterprise
### Author: DevOps Engineering Team
### Date: August 2025

---

## Overview

This deployment guide covers the Entertainment Investment Intelligence Platform deployment across development, staging, and production environments with focus on high-availability analytics and machine learning workloads.

## Prerequisites

### System Requirements

**Production Requirements**:
- CPU: 16 cores (Intel Xeon or AMD EPYC)
- RAM: 128GB DDR4 ECC memory
- Storage: 2TB NVMe SSD
- Network: 10Gbps connectivity
- GPU: Optional NVIDIA V100/A100 for ML acceleration

### Software Dependencies

- **Runtime**: Python 3.11+, Node.js 18+
- **Containers**: Docker 24.0+, Kubernetes 1.28+
- **Databases**: PostgreSQL 15+, MongoDB 6.0+, Redis 7+
- **ML Stack**: TensorFlow 2.13+, PyTorch 2.0+, Scikit-learn 1.3+

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/enterprise/entertainment-intelligence.git
cd entertainment-intelligence

# Create virtual environment
python3.11 -m venv entertainment_env
source entertainment_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Database Configuration

```bash
# PostgreSQL setup
sudo apt install postgresql-15
sudo systemctl start postgresql

# Create entertainment database
sudo -u postgres createdb entertainment_intelligence
```

```sql
-- Database optimization
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
SELECT pg_reload_conf();
```

### 3. Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://entertainment:password@localhost/entertainment_intelligence
MONGODB_URL=mongodb://localhost:27017/entertainment
REDIS_URL=redis://localhost:6379

# ML Configuration
MODEL_PATH=./models
FEATURE_STORE_PATH=./data/features
GPU_ENABLED=false
BATCH_SIZE=32

# External APIs
TMDB_API_KEY=your-tmdb-key
IMDB_API_KEY=your-imdb-key
TWITTER_API_KEY=your-twitter-key
YOUTUBE_API_KEY=your-youtube-key

# Security
JWT_SECRET_KEY=your-jwt-secret-32-characters
ENCRYPTION_KEY=your-encryption-key-32-chars

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
EOF
```

### 4. Start Development Environment

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Run application
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Verify health
curl http://localhost:8000/health
```

## Docker Deployment

### 1. Production Dockerfile

```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS production

RUN groupadd -r entertainment && useradd -r -g entertainment entertainment

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/entertainment/.local

# Copy application
COPY --chown=entertainment:entertainment Files/ ./Files/
COPY --chown=entertainment:entertainment models/ ./models/
COPY --chown=entertainment:entertainment *.py ./

ENV PATH=/home/entertainment/.local/bin:$PATH
ENV PYTHONPATH=/app

USER entertainment

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Docker Compose Configuration

```yaml
version: '3.8'

services:
  entertainment-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://entertainment:password@postgres:5432/entertainment_intelligence
      - MONGODB_URL=mongodb://mongo:27017/entertainment
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - postgres
      - mongo
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: entertainment_intelligence
      POSTGRES_USER: entertainment
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  mongo:
    image: mongo:6.0
    environment:
      MONGO_INITDB_DATABASE: entertainment
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 4gb
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  mongo_data:
  redis_data:
```

## Kubernetes Deployment

### 1. Application Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: entertainment-app
  namespace: entertainment-intelligence
spec:
  replicas: 3
  selector:
    matchLabels:
      app: entertainment-app
  template:
    metadata:
      labels:
        app: entertainment-app
    spec:
      containers:
      - name: entertainment-app
        image: enterprise/entertainment-intelligence:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: entertainment-secrets
              key: database-url
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### 2. Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: entertainment-service
  namespace: entertainment-intelligence
spec:
  selector:
    app: entertainment-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Cloud Deployment

### 1. AWS EKS Setup

```bash
# Create EKS cluster
eksctl create cluster \
  --name entertainment-cluster \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name entertainment-nodes \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name entertainment-cluster
```

### 2. Database Setup

```bash
# Create RDS PostgreSQL
aws rds create-db-instance \
  --db-instance-identifier entertainment-postgres \
  --db-instance-class db.r5.2xlarge \
  --engine postgres \
  --engine-version 15.3 \
  --master-username entertainment \
  --allocated-storage 500 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-entertainment \
  --multi-az \
  --storage-encrypted

# Create DocumentDB (MongoDB-compatible)
aws docdb create-db-cluster \
  --db-cluster-identifier entertainment-docdb \
  --engine docdb \
  --master-username entertainment \
  --master-user-password SecurePassword123 \
  --vpc-security-group-ids sg-entertainment
```

## Performance Optimization

### 1. Application Tuning

```python
# Gunicorn configuration
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 5000
timeout = 120
keepalive = 2
preload_app = True
```

### 2. Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

## Monitoring & Alerting

### 1. Prometheus Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'entertainment-app'
    static_configs:
      - targets: ['entertainment-service:8000']
    metrics_path: /metrics
```

### 2. Critical Alerts

```yaml
groups:
- name: entertainment-alerts
  rules:
  - alert: HighPredictionLatency
    expr: prediction_latency_p95 > 1000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Prediction latency high"
      
  - alert: ModelAccuracyDrop
    expr: model_accuracy < 0.85
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Model accuracy below threshold"
```

## Security Configuration

### 1. Network Security

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: entertainment-network-policy
spec:
  podSelector:
    matchLabels:
      app: entertainment-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

### 2. Secrets Management

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: entertainment-secrets
type: Opaque
stringData:
  database-url: "postgresql://user:pass@host:5432/db"
  tmdb-api-key: "your-tmdb-api-key"
  jwt-secret: "your-jwt-secret-key"
```

## Backup & Recovery

### 1. Database Backup

```bash
#!/bin/bash
# Backup script
BACKUP_DIR="/backups/entertainment"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump -h $POSTGRES_HOST -U entertainment -d entertainment_intelligence \
  > "$BACKUP_DIR/postgres_backup_$DATE.sql"

# MongoDB backup  
mongodump --host $MONGO_HOST --db entertainment \
  --out "$BACKUP_DIR/mongo_backup_$DATE"

# Upload to S3
aws s3 cp "$BACKUP_DIR/" s3://entertainment-backups/ --recursive
```

### 2. Model Backup

```bash
# Model backup and versioning
tar -czf "models_backup_$(date +%Y%m%d).tar.gz" ./models/
aws s3 cp "models_backup_$(date +%Y%m%d).tar.gz" s3://entertainment-models/
```

This deployment guide provides comprehensive instructions for deploying the Entertainment Investment Intelligence Platform with enterprise-grade reliability, security, and performance.