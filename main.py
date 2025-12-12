from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import faiss

# Cliente de OpenAI (usa la variable de entorno OPENAI_API_KEY en Render)
client = OpenAI()

app = FastAPI(title="Wise Lucid Oracle API")

# ---- CORS para que tu web (Shopify u otra) pueda llamar a esta API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # luego podemos limitarlo a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- MODELO DE DATOS ----------------

class OracleQuestion(BaseModel):
    question: str

# --------------- VECTOR STORE / EMBEDDINGS ----------------

EMBED_DIM = 1536  # dimensión de text-embedding-3-small
index = faiss.IndexFlatL2(EMBED_DIM)

oracle_texts = [
    "Each thought is a doorway into your own awareness.",
    "Clarity appears when you allow inner noise to rest.",
    "Wisdom is not created, it is remembered. You already carry it within."
]

def embed(text: str) -> np.ndarray:
    """Genera un embedding para el texto dado usando OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Pre-calculamos embeddings de las frases base del oráculo
oracle_vectors = [embed(t) for t in oracle_texts]
index.add(np.array(oracle_vectors))

# --------------- ENDPOINT PRINCIPAL DEL ORÁCULO ----------------

@app.post("/api/oracle")
def oracle_answer(payload: OracleQuestion):

    q = payload.question.strip()

    # Si la pregunta viene vacía, devolvemos un mensaje amable
    if not q:
        return {
            "oracle_message": (
                "The oracle needs at least a whisper from your heart to respond. "
                "Write something that truly matters to you."
            ),
            "related_phrase": None
        }

    # Embed de la pregunta
    q_vec = embed(q).reshape(1, -1)

    # Buscar la frase más cercana en el espacio vectorial
    D, I = index.search(q_vec, 1)
    selected_text = oracle_texts[I[0][0]]

    # Llamada al modelo con instrucciones de idioma
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Wise Lucid Oracle. "
                    "Your tone is poetic, gentle, philosophical, and compassionate. "
                    "You never predict the future or give orders; you invite reflection "
                    "and help the user remember their inner wisdom. "
                    "IMPORTANT: First, detect the language of the user's question. "
                    "Then respond ENTIRELY in that same language. "
                    "If the user writes in English, answer only in English. "
                    "If the user writes in Spanish, answer only in Spanish. "
                    "Do not mix languages in a single answer. "
                    "Keep your replies brief, clear, and emotionally supportive."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {q}\n\n"
                    f"Use this phrase only as inner inspiration "
                    f"(you DO NOT need to repeat it literally): '{selected_text}'.\n"
                    "Answer briefly, poetically, and deeply, following the rules above."
                )
            }
        ]
    )

    oracle_message = completion.choices[0].message.content

    return {
        "oracle_message": oracle_message,
        "related_phrase": selected_text
    }

# --------------- ENDPOINT DE PRUEBA ----------------

@app.get("/")
def home():
    return {"message": "Wise Lucid Oracle is running ✨"}
