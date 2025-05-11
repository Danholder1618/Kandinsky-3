# inference_service/Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ffmpeg \
    wget \
    && wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml \
       -P /usr/local/share/opencv4/haarcascades/ \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt exact_requirements.txt setup.py ./

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r exact_requirements.txt \
 && pip install opencv-python-headless

# Копируем код
COPY kandinsky3 ./kandinsky3
COPY inference_service/app.py ./

# Устанавливаем пакет
RUN pip install -e .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]