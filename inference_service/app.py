from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import cv2
import numpy as np

# Импорт из Kandinsky-3
from kandinsky3 import get_inpainting_pipeline

app = FastAPI(title="Kandinsky-3 Selfie Style Service")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Настройка типов данных
dtype_map = {
    'unet': torch.float16 if device == "cuda" else torch.float32,
    'text_encoder': torch.float16 if device == "cuda" else torch.float32,
    'movq': torch.float32,
}

# Предзагрузка пайплайна для инпейнтинга
print("Loading Kandinsky-3 inpainting pipeline...")
inp_pipe = get_inpainting_pipeline(device, dtype_map)
print("Pipeline loaded successfully!")

def create_face_mask(image_pil, padding_percent=0.2):
    """
    Создает маску для селфи, где лицо сохраняется (0), а фон изменяется (1).
    Использует OpenCV для детекции лица.
    """
    # Конвертируем PIL -> OpenCV
    image = np.array(image_pil.convert('RGB'))
    image_rgb = image.copy()
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Загрузка каскадного классификатора Haar для обнаружения лиц
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    except Exception as e:
        print(f"Error in face detection: {e}")
        faces = []
    
    # Создаем маску: 1 - для изменения (фон), 0 - для сохранения (лицо)
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if len(faces) > 0:
        # Если лица найдены, отмечаем их на маске
        for (x, y, w, h) in faces:
            # Добавляем отступ для лучшего сохранения лица
            padding = int(max(w, h) * padding_percent)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            # Устанавливаем область лица как 0 (сохраняемая область)
            mask[y_start:y_end, x_start:x_end] = 0
    else:
        # Если лицо не найдено, используем центр изображения
        print("No faces detected, using center of image")
        height, width = mask.shape
        center_h, center_w = height // 2, width // 2
        face_size = min(height, width) // 3  # Примерный размер лица
        
        y_start = max(0, center_h - face_size)
        y_end = min(height, center_h + face_size)
        x_start = max(0, center_w - face_size)
        x_end = min(width, center_w + face_size)
        
        mask[y_start:y_end, x_start:x_end] = 0
    
    return mask

def resize_if_needed(image, max_dim=1024):
    """Масштабирует изображение, если оно слишком большое"""
    w, h = image.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

@app.post("/generate/")
async def generate_styled_selfie(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    padding: float = Form(0.2)  # Дополнительный отступ вокруг лица
):
    try:
        # Читаем изображение
        data = await file.read()
        selfie = Image.open(io.BytesIO(data)).convert("RGB")
        
        # Масштабируем, если нужно
        selfie = resize_if_needed(selfie)
        
        # Создаем маску для лица
        mask = create_face_mask(selfie, padding_percent=padding)
        
        print(f"Processing with prompt: '{prompt}', image size: {selfie.size}")
        
        # Генерируем новый фон с помощью Kandinsky
        result = inp_pipe(prompt, selfie, mask).images[0]
        
        # Возвращаем результат
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/debug/mask/")
async def debug_mask(
    file: UploadFile = File(...),
    padding: float = Form(0.2)
):
    """Эндпоинт для отладки - показывает маску, которая будет использоваться"""
    try:
        data = await file.read()
        selfie = Image.open(io.BytesIO(data)).convert("RGB")
        selfie = resize_if_needed(selfie)
        
        # Создаем маску
        mask = create_face_mask(selfie, padding_percent=padding)
        
        # Преобразуем маску в изображение (для визуализации)
        mask_img = Image.fromarray(mask * 255)  # 0->0 (черный), 1->255 (белый)
        
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mask generation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Простая проверка работоспособности сервиса"""
    return {"status": "ok", "device": device}