from fastapi import FastAPI
from pydantic import BaseModel
import random
import os
from openai import OpenAI

app = FastAPI()

# Inicializar cliente con tu API key desde las variables de entorno
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Lista de frases inspiradoras
INSPIRATIONAL_PHRASES = [
    "La sabiduría no se crea, se recuerda.",
    "Todo lo que buscas ya te habita en silencio.",
    "La luz se reconoce más profundamente en la sombra.",
    "El corazón comprende lo que la mente no alcanza.",
    "Solo lo que aceptas te transforma.",
    "La intuición es la voz suave de tu propio camino.",
    "El presente es el único lugar donde algo real ocurre.",
    "Lo que eres nunca ha estado separado de lo que anhelas."
]

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Wise Lucid Oracle is running ✨"}

@app.post("/api/oracle")
def oracle_answer(data: Question):

    user_question = data.question.strip()

    # Seleccionar una frase inspiradora al azar
    selected_text = random.choice(INSPIRATIONAL_PHRASES)

    # Llamada al modelo
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Wise Lucid Oracle. "
                    "Your tone is poetic, soft, philosophical and compassionate. "
                    "You do not predict the future or give commands. "
                    "You inspire and help the user remember their inner wisdom. "
                    "VERY IMPORTANT: Always respond in the SAME LANGUAGE the user writes in "
                    "(if the question is in English, answer in English; if it's in Spanish, answer in Spanish)."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_question}\n\n"
                    f"Use this as inner inspiration (DO NOT repeat it literally): '{selected_text}'.\n"
                    "Answer briefly, poetically, and deeply. "
                    "Keep the same language as the question."
                )
            }
        ]
    )

    oracle_message = completion.choices[0].message["content"]

    return {
        "oracle_message": oracle_message,
        "related_phrase": selected_text
    }
