FROM python:3.11-slim

WORKDIR /app

# 1. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install system dependencies
# FIX: Replaced 'libgl1-mesa-glx' with 'libgl1' for compatibility with newer Debian versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libjpeg62-turbo \
    libfreetype6 \
    liblcms2-2 \
    libopenjp2-7 \
    libtiff6 \
    libwebp7 \
    tcl \
    tk \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run database migrations and collect static files
RUN python manage.py makemigrations && \
    python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run application
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]