# TreeRAG Docker Deployment Guide

## 🐳 Quick Start with Docker

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Gemini API Key

### 1. Setup Environment
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### 2. Build and Run
```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Access Services
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📦 Production Deployment

### Environment Variables
```bash
# Backend
GOOGLE_API_KEY=your_api_key
USE_DEEP_TRAVERSAL=true
MAX_TRAVERSAL_DEPTH=5
MAX_BRANCHES_PER_LEVEL=3

# Frontend
NEXT_PUBLIC_API_BASE_URL=http://your-backend-url/api
```

### Build for Production
```bash
# Build images
docker-compose build --no-cache

# Tag for registry
docker tag treerag-backend:latest your-registry/treerag-backend:v1.0
docker tag treerag-frontend:latest your-registry/treerag-frontend:v1.0

# Push to registry
docker push your-registry/treerag-backend:v1.0
docker push your-registry/treerag-frontend:v1.0
```

### Scale Services
```bash
# Scale backend for high load
docker-compose up -d --scale backend=3
```

## 🔧 Development with Docker

```bash
# Run tests in container
docker-compose exec backend pytest tests/

# Shell access
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🛠 Troubleshooting

### Container won't start
```bash
docker-compose logs backend
docker-compose logs frontend
```

### Clear cache and rebuild
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Check health
```bash
docker-compose ps
docker inspect --format='{{.State.Health}}' treerag-backend
```

## 📊 Monitoring

### Resource Usage
```bash
docker stats
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend --tail=100
```
