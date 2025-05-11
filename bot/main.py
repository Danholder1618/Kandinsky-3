import os
import io
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Command, InlineKeyboardMarkup, InlineKeyboardButton
from aiohttp import ClientSession, FormData
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_URL      = os.getenv("MODEL_URL")  # http://model:8000

bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher()

# Состояния пользователя
# waiting_for_photo → waiting_for_style → waiting_for_prompt → finished
user_states = {}
user_images = {}
user_styles = {}

# Предопределённые стили
STYLES = ["Van Gogh", "Picasso", "Synthwave", "Steampunk", "Cyberpunk"]

def style_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton(text=s, callback_data=f"choose_style:{s}")
        for s in STYLES
    ]
    kb.add(*buttons)
    return kb

def postgen_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
      InlineKeyboardButton("↺ Переделать", callback_data="redo"),
      InlineKeyboardButton("📸 Новое фото", callback_data="new_photo")
    )
    return kb

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    # Сбросим всё
    user_states[uid] = "waiting_for_photo"
    user_images.pop(uid, None)
    user_styles.pop(uid, None)

    await message.answer(
        "Привет! Пришли мне своё селфи, чтобы начать стилизацию.",
        reply_markup=None
    )

@dp.message(content_types=types.ContentType.PHOTO)
async def photo_handler(message: types.Message):
    uid = message.from_user.id
    if user_states.get(uid) not in ("waiting_for_photo", "finished"):
        return await message.answer("Нажми /start, чтобы начать заново.")

    file = await bot.get_file(message.photo[-1].file_id)
    data = await bot.download_file(file.file_path)
    user_images[uid] = data.getvalue()
    user_states[uid] = "waiting_for_style"

    await message.answer(
        "Селфи получено! Выбери стиль из списка или пришли свой текст:",
        reply_markup=style_keyboard()
    )

@dp.callback_query(F.data.startswith("choose_style:"))
async def style_chosen(call: types.CallbackQuery):
    uid = call.from_user.id
    if user_states.get(uid) != "waiting_for_style":
        return await call.answer()
    style = call.data.split(":",1)[1]
    user_styles[uid] = style
    user_states[uid] = "waiting_for_prompt"

    await call.message.answer(
        f"Стиль «{style}» выбран. Теперь пришли текст, как переделать селфи.",
    )
    await call.answer()

@dp.message(lambda m: m.text and user_states.get(m.from_user.id) == "waiting_for_style")
async def custom_style(message: types.Message):
    uid = message.from_user.id
    # текст вместо кнопки
    user_styles[uid] = message.text
    user_states[uid] = "waiting_for_prompt"
    await message.answer(f"Стиль «{message.text}» сохранён. Пришли текст для генерации.")

@dp.message(lambda m: m.text and user_states.get(m.from_user.id) == "waiting_for_prompt")
async def prompt_handler(message: types.Message):
    uid    = message.from_user.id
    prompt = message.text
    img    = user_images.get(uid)
    style  = user_styles.get(uid)
    if not img or not style:
        return await message.answer("Что-то пошло не так, отправь /start и начни заново.")

    full_prompt = f"{style}, {prompt}".strip(", ")
    await message.answer("Генерирую…")

    # Отправляем в inference-сервис
    async with ClientSession() as sess:
        form = FormData()
        form.add_field("prompt", full_prompt)
        form.add_field("file", img, filename="selfie.png", content_type="image/png")
        resp = await sess.post(f"{MODEL_URL}/generate/", data=form)
        if resp.status != 200:
            return await message.answer("❌ Ошибка генерации, попробуй позже.")
        out = await resp.read()

    await message.answer_photo(
        out,
        caption="🎨 Вот твоё селфи в новом стиле!",
        reply_markup=postgen_keyboard()
    )
    user_states[uid] = "finished"

@dp.callback_query(F.data == "redo")
async def redo_handler(call: types.CallbackQuery):
    uid = call.from_user.id
    if user_states.get(uid) != "finished":
        return await call.answer()
    user_states[uid] = "waiting_for_style"
    await call.message.reply("Окей, выбери другой стиль:", reply_markup=style_keyboard())
    await call.answer()

@dp.callback_query(F.data == "new_photo")
async def newphoto_handler(call: types.CallbackQuery):
    uid = call.from_user.id
    user_states[uid] = "waiting_for_photo"
    user_images.pop(uid, None)
    user_styles.pop(uid, None)
    await call.message.reply("Хорошо, пришли новое селфи.")
    await call.answer()

@dp.message()
async def fallback(message: types.Message):
    uid   = message.from_user.id
    state = user_states.get(uid)
    if state == "waiting_for_photo":
        await message.answer("Пожалуйста, отправь своё селфи, чтобы начать.")
    elif state == "waiting_for_style":
        await message.answer("Выбери стиль кнопкой или пришли свой текст.")
    elif state == "waiting_for_prompt":
        await message.answer("Теперь пришли текстовое описание.")
    else:
        await message.answer("Нажми /start, чтобы начать заново.")

if __name__ == "__main__":
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
