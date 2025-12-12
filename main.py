from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import faiss

from fastapi.middleware.cors import CORSMiddleware

# Inicializa cliente OpenAI (usa OPENAI_API_KEY de Render)
client = OpenAI()

app = FastAPI(title="Wise Lucid Oracle API")

# ---- CORS: permite que Shopify llame a la API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego podemos limitarlo a tu dominio
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

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Pre-cálculo de embeddings
oracle_vectors = [embed(t) for t in oracle_texts]
index.add(np.array(oracle_vectors))

# --------------- ORACLE RESPONSE ----------------

@app.post("/api/oracle")
def oracle_answer(payload: OracleQuestion):

    q = payload.question

    # Embed de la pregunta
    q_vec = embed(q).reshape(1, -1)

    # Busca la frase más cercana
    D, I = index.search(q_vec, 1)
    selected_text = oracle_texts[I[0][0]]

    # Respuesta estilo Wise Lucid
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el Oráculo de Wise Lucid. "
                    "Respondes con un tono poético, suave, filosófico, "
                    "sin predecir el futuro ni dar órdenes. "
                    "Tu misión es inspirar, no dirigir."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta del usuario: {q}\n\n"
                    f"Inspírate en esta frase: '{selected_text}'. "
                    "Responde con una reflexión breve y profunda."
                )
            }
        ]
    )

    answer = completion.choices[0].message.content

    return {
        "oracle_message": answer,
        "related_phrase": selected_text
    }

# --------------- ROOT TEST ----------------

@app.get("/")
def home():
    return {"message": "Wise Lucid Oracle is running ✨"}
