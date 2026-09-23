# ---------- FRONTEND BUILD ----------
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# ---------- BACKEND ----------
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/backend/requirements.txt

# Backend code
COPY backend/ /app/backend/

# Built React frontend
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Railway PORT
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "backend/main_database_camera_fixed.py"]