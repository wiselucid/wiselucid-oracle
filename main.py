from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import faiss

# Cliente de OpenAI (usa OPENAI_API_KEY de Render)
client = OpenAI()

app = FastAPI(title="Wise Lucid Oracle API")

# ---- CORS para que Shopify u otras webs puedan llamar a la API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # luego podemos limitarlo a tu dominio de Shopify
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- DATA MODEL ----------------

class OracleQuestion(BaseModel):
    question: str

# --------------- VECTOR STORE SETUP ----------------

EMBED_DIM = 1536  # para text-embedding-3-small
index = faiss.IndexFlatL2(EMBED_DIM)

oracle_texts = [
    "Cada pensamiento es una puerta hacia tu conciencia.",
    "La claridad surge cuando permites que el ruido interno descanse.",
    "La sabiduría no se crea, se recuerda. Tú ya la llevas dentro."
]

def embed(text: str) -> np.ndarray:
    """Genera un embedding para el texto dado."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Pre-calculamos embeddings de las frases del oráculo
oracle_vectors = [embed(t) for t in oracle_texts]
index.add(np.array(oracle_vectors))

# --------------- ORACLE RESPONSE ----------------

@app.post("/api/oracle")
def oracle_answer(payload: OracleQuestion):

    q = payload.question.strip()

    # Si la pregunta viene vacía, devolvemos un mensaje amable
    if not q:
        return {
            "oracle_message": "El oráculo necesita al menos un susurro de tu corazón para responder. Escribe algo que realmente te importe.",
            "related_phrase": None
        }

    # Embed de la pregunta
    q_vec = embed(q).reshape(1, -1)

    # Buscar la frase más cercana
    D, I = index.search(q_vec, 1)
    selected_text = oracle_texts[I[0][0]]

    # Crear respuesta estilo Wise Lucid
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Wise Lucid Oracle. "
                    "Your tone is poetic, soft, philosophical and compassionate. "
                    "You do not predict the future or give commands; "
                    "you invite reflection and help the user remember their inner wisdom. "
                    "VERY IMPORTANT: Always respond in the SAME LANGUAGE the user writes in "
                    "(if the question is in English, answer in English; if it's in Spanish, answer in Spanish)."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {q}\n\n"
                    f"Use this phrase only as inner inspiration (you DO NOT need to repeat it literally): '{selected_text}'.\n"
                    "Answer briefly, poetically, and deeply. "
                    "Keep the same language as the question."
                )
            }
        ]
    )

    oracle_message = completion.choices[0].message.content

    return {
        "oracle_message": oracle_message,
        "related_phrase": selected_text
    }

# --------------- ROOT TEST ----------------

@app.get("/")
def home():
    return {"message": "Wise Lucid Oracle is running ✨"}
