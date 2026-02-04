# Security Best Practices

## 환경 변수 보안

### ⚠️ 현재 문제점

docker-compose.yml에서 환경 변수를 평문으로 전달하면:
- `docker-compose logs`에 노출
- `docker inspect`에 노출  
- 컨테이너 내부에서 `env` 명령으로 확인 가능

### ✅ 해결 방법

#### 방법 1: Docker Secrets 사용 (권장)

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    secrets:
      - google_api_key
    environment:
      - GOOGLE_API_KEY_FILE=/run/secrets/google_api_key
    # ... 나머지 설정

secrets:
  google_api_key:
    file: ./secrets/google_api_key.txt
```

```bash
# secrets 디렉토리 생성
mkdir -p secrets
echo "YOUR_API_KEY_HERE" > secrets/google_api_key.txt
chmod 600 secrets/google_api_key.txt

# .gitignore에 추가
echo "secrets/" >> .gitignore
```

```python
# src/config.py 수정
api_key_file = os.getenv("GOOGLE_API_KEY_FILE")
if api_key_file:
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
else:
    api_key = os.getenv("GOOGLE_API_KEY")
```

#### 방법 2: .env 파일 사용 (현재 방식 개선)

```bash
# .env 파일 권한 제한
chmod 600 .env

# Docker Compose 실행 시 주의사항
docker-compose config  # ❌ 환경변수 노출됨
docker-compose up      # ✅ 환경변수 숨겨짐
```

#### 방법 3: 외부 시크릿 관리 시스템

프로덕션 환경에서는:
- **AWS Secrets Manager**
- **HashiCorp Vault**
- **Azure Key Vault**
- **Google Secret Manager**

```python
# 예시: AWS Secrets Manager
import boto3
import json

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )
    
    response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response['SecretString'])
    return secret['GOOGLE_API_KEY']

api_key = get_secret('treerag/api-keys')
```

## 로깅 보안

### ❌ 나쁜 예

```python
logger.info(f"API Key: {api_key}")
logger.error(f"Failed with key: {api_key[:10]}...")
print(f"Using key: {api_key}")
```

### ✅ 좋은 예

```python
logger.info("API authentication configured")
logger.error("API authentication failed")

# 디버깅 필요 시 마스킹
masked_key = f"{api_key[:4]}...{api_key[-4:]}" if api_key else "None"
logger.debug(f"API Key (masked): {masked_key}")
```

## 에러 메시지 보안

### ❌ 나쁜 예

```python
raise ValueError(f"Missing GOOGLE_API_KEY in .env file")
raise HTTPException(detail=f"Database connection failed: {conn_string}")
```

### ✅ 좋은 예

```python
logger.error("Configuration error: missing API key")
raise ValueError("Configuration error: missing required environment variable")

logger.error(f"Database error: {type(e).__name__}")
raise HTTPException(
    status_code=500,
    detail="Internal server error"
)
```

## 프로세스 메모리 보호

```bash
# 컨테이너 보안 강화
# docker-compose.yml
services:
  backend:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
      - /var/cache
```

## 감사 로그

```python
# src/utils/audit_logger.py
import logging
from datetime import datetime

audit_logger = logging.getLogger('audit')

def log_sensitive_operation(operation: str, user_id: str, success: bool):
    """민감한 작업 감사 로그"""
    audit_logger.info({
        "timestamp": datetime.utcnow().isoformat(),
        "operation": operation,
        "user_id": user_id,
        "success": success,
        "ip": "masked_for_privacy"
    })

# 사용 예시
log_sensitive_operation("api_key_rotation", "admin", True)
```

## 체크리스트

프로덕션 배포 전:

- [ ] API 키를 Docker Secrets로 이동
- [ ] .env 파일 권한 `chmod 600` 설정
- [ ] 에러 메시지에서 민감 정보 제거
- [ ] 로그에 API 키/비밀번호 출력 금지
- [ ] 컨테이너 read-only 파일시스템 적용
- [ ] 감사 로깅 활성화
- [ ] 시크릿 로테이션 정책 수립
- [ ] 접근 제어 (최소 권한 원칙)

## 추가 리소스

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [12-Factor App: Config](https://12factor.net/config)
