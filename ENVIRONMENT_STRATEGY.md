# Environment Strategy

## Three-Tier Environment Model

### 1. Local Development & Testing
- **Purpose**: Development and initial testing on the developer's machine or CI.
- **Tools**: Python, Flask, Docker, pytest, flake8.
- **Configuration Files**: `.env.development`.
- **Environment Variables**:
  - `FLASK_ENV=development`
  - `DATABASE_URL=sqlite:///dev.db`
- **Setup**: Docker Compose for local services.

### 2. Pre-Production
- **Purpose**: Staging environment for end-to-end validation.
- **Branch**: `preprod`.
- **Tools**: Docker, Terraform, GitHub Actions.
- **Configuration Files**: `.env.preprod`.
- **Environment Variables**:
  - `FLASK_ENV=staging`
  - `DATABASE_URL=postgres://user:password@staging-db:5432/db`
- **Setup**: Deploy to a staging server using Terraform.

### 3. Production
- **Purpose**: Live deployment.
- **Branch**: `main`.
- **Tools**: Docker, Terraform, GitHub Actions.
- **Configuration Files**: `.env.production`.
- **Environment Variables**:
  - `FLASK_ENV=production`
  - `DATABASE_URL=postgres://user:password@prod-db:5432/db`
- **Setup**: Deploy to production server using Terraform.

---

## Required Tools
- **Python**: For application development.
- **Flask**: Web framework.
- **Docker**: Containerization.
- **Terraform**: Infrastructure as code.
- **GitHub Actions**: CI/CD pipelines.

---

## Configuration Files
- `.env.development`: Local development environment variables.
- `.env.preprod`: Pre-production environment variables.
- `.env.production`: Production environment variables.

---

## Commit Message
```
chore: define three-tier environment strategy
```
