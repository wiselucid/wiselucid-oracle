import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
import faiss
import numpy as np

# =====================
# 1. CONFIGURACIÓN OPENAI
# =====================

os.environ["OPENAI_API_KEY"] = "AQUI_TU_API_KEY"   # <- reemplaza esto

client = OpenAI()

# =====================
# 2. DATASET Q&A (30 PREGUNTAS)
# =====================

qa_pairs = [
    {
        "id": 1,
        "question": "No sé cuál es mi propósito. ¿Qué hago?",
        "answer": """El propósito aparece cuando comienzas a escuchar lo que te enciende por dentro. 
No busques una misión escrita; siente las pequeñas chispas que ya están en ti. 
Camina hacia ellas con honestidad.  
Esencia simbólica: Wild Oat  
Cristal simbólico: Labradorita"""
    },
    {
        "id": 2,
        "question": "Me siento perdido con mi vida. Nada tiene sentido.",
        "answer": """Cuando te sientes perdido es porque una forma antigua de ti ya no sirve. 
No te presiones: lo nuevo necesita silencio para revelarse.  
Esencia simbólica: Cerato  
Cristal simbólico: Cuarzo ahumado"""
    },
    {
        "id": 3,
        "question": "¿Qué va a pasar conmigo en el futuro?",
        "answer": """No puedo ver tu futuro, pero sí puedo recordarte que el futuro nace de cómo habitas tu presente. 
Cuida tu respiración, tus decisiones pequeñas y tu honestidad contigo mismo.  
Esencia simbólica: Rescue Remedy  
Cristal simbólico: Amatista"""
    },
    {
        "id": 4,
        "question": "¿Cómo dejo de sentir este dolor que llevo dentro?",
        "answer": """El dolor es un mensajero, no un castigo. Cuando lo observas con suavidad, comienza a transformarse.  
Esencia simbólica: Star of Bethlehem  
Cristal simbólico: Cuarzo rosa"""
    },
    {
        "id": 5,
        "question": "Me rompieron el corazón. ¿Cómo sigo adelante?",
        "answer": """Un corazón roto es una grieta por donde entra la luz. Deja que duela sin prisa.  
Esencia simbólica: Honeysuckle  
Cristal simbólico: Rodonita"""
    },
    {
        "id": 6,
        "question": "No tengo energía para nada. ¿Qué me pasa?",
        "answer": """A veces tu cuerpo te pide una pausa sagrada. Escucha antes de apresurarte.  
Esencia simbólica: Olive  
Cristal simbólico: Citrino"""
    },
    {
        "id": 7,
        "question": "Me siento solo incluso rodeado de gente.",
        "answer": """La soledad interna te invita a regresar a ti. Acompáñate con honestidad.  
Esencia simbólica: Water Violet  
Cristal simbólico: Howlita"""
    },
    {
        "id": 8,
        "question": "¿Esta persona es para mí?",
        "answer": """Pregúntate si en su presencia te expandes o te encoges. Esa respuesta ya sabe.  
Esencia simbólica: Walnut  
Cristal simbólico: Kunzita"""
    },
    {
        "id": 9,
        "question": "¿Debo cambiar de trabajo?",
        "answer": """Escucha si tu impulso nace de huida o de crecimiento. Eso lo cambia todo.  
Esencia simbólica: Scleranthus  
Cristal simbólico: Ojo de tigre"""
    },
    {
        "id": 10,
        "question": "Siento que he fallado en la vida.",
        "answer": """El fracaso es una historia escrita desde la dureza. Reescribe desde la compasión.  
Esencia simbólica: Pine  
Cristal simbólico: Ópalo blanco"""
    },
    {
        "id": 11,
        "question": "¿Cómo puedo confiar más en mí?",
        "answer": """La confianza crece con pequeñas promesas cumplidas contigo.  
Esencia simbólica: Larch  
Cristal simbólico: Sodalita"""
    },
    {
        "id": 12,
        "question": "Me da miedo cambiar mi vida. ¿Y si sale mal?",
        "answer": """El miedo no desaparece: se acompaña. Camina con él, no contra él.  
Esencia simbólica: Mimulus  
Cristal simbólico: Cornalina"""
    },
    {
        "id": 13,
        "question": "Estoy confundido. No sé qué quiero.",
        "answer": """La claridad aparece cuando baja el ruido. Siente, no forces.  
Esencia simbólica: Clematis  
Cristal simbólico: Fluorita"""
    },
    {
        "id": 14,
        "question": "Todos avanzan menos yo.",
        "answer": """No compares tu proceso interno con los resultados externos de otros.  
Esencia simbólica: Crab Apple  
Cristal simbólico: Aguamarina"""
    },
    {
        "id": 15,
        "question": "Me siento bloqueado creativamente.",
        "answer": """La creatividad regresa cuando la dejas jugar.  
Esencia simbólica: Wild Rose  
Cristal simbólico: Amazonita"""
    },
    {
        "id": 16,
        "question": "¿Cómo dejo de exigirme tanto?",
        "answer": """Trátate como tratarías a alguien que amas: con límites y ternura.  
Esencia simbólica: Vervain  
Cristal simbólico: Calcita naranja"""
    },
    {
        "id": 17,
        "question": "Mi vida se siente detenida.",
        "answer": """A veces afuera nada se mueve porque adentro todo está cambiando.  
Esencia simbólica: Hornbeam  
Cristal simbólico: Obsidiana dorada"""
    },
    {
        "id": 18,
        "question": "¿Cómo conecto con algo más grande?",
        "answer": """Lo grande empieza en ti. Respira y observa tu mundo interno.  
Esencia simbólica: Aspen  
Cristal simbólico: Selenita"""
    },
    {
        "id": 19,
        "question": "Me saboteo todo el tiempo. ¿Por qué?",
        "answer": """Una parte de ti intenta protegerte de lo desconocido. Agradécele y redirige.  
Esencia simbólica: Chicory  
Cristal simbólico: Hematita"""
    },
    {
        "id": 20,
        "question": "No sé quién soy.",
        "answer": """No saber te libera para volver a elegir quién deseas ser ahora.  
Esencia simbólica: Walnut  
Cristal simbólico: Piedra luna"""
    },
    {
        "id": 21,
        "question": "¿Cómo sé la decisión correcta?",
        "answer": """La decisión correcta no siempre es ligera; a veces solo es honesta.  
Esencia simbólica: Scleranthus  
Cristal simbólico: Jaspe rojo"""
    },
    {
        "id": 22,
        "question": "¿Cómo puedo quererme más?",
        "answer": """El amor propio se construye con pequeñas acciones hacia ti.  
Esencia simbólica: Gentian  
Cristal simbólico: Cuarzo rosa"""
    },
    {
        "id": 23,
        "question": "Me da miedo que no me quieran.",
        "answer": """El rechazo no define tu valor; define la falta de resonancia.  
Esencia simbólica: Heather  
Cristal simbólico: Turmalina rosa"""
    },
    {
        "id": 24,
        "question": "Me arrepiento de decisiones del pasado.",
        "answer": """Tu yo del pasado hizo lo mejor que pudo con lo que sabía.  
Esencia simbólica: Pine  
Cristal simbólico: Jade"""
    },
    {
        "id": 25,
        "question": "Siento un vacío que no sé llenar.",
        "answer": """El vacío no es carencia: es espacio para algo nuevo.  
Esencia simbólica: Mustard  
Cristal simbólico: Ágata blanca"""
    },
    {
        "id": 26,
        "question": "¿Cuál es el sentido de todo esto?",
        "answer": """El sentido se construye con tus decisiones pequeñas.  
Esencia simbólica: Gorse  
Cristal simbólico: Cianita"""
    },
    {
        "id": 27,
        "question": "¿Me va a ir bien si tomo esta decisión?",
        "answer": """No veo tu futuro, pero sí tu presente. Actúa desde la coherencia.  
Esencia simbólica: Red Chestnut  
Cristal simbólico: Labradorita"""
    },
    {
        "id": 28,
        "question": "No sé vivir sin esta persona.",
        "answer": """No confundas amor con identidad. Regresa a ti con suavidad.  
Esencia simbólica: Centaury  
Cristal simbólico: Cuarzo verde"""
    },
    {
        "id": 29,
        "question": "Dime exactamente qué hacer.",
        "answer": """No decido por ti; te ayudo a escucharte mejor.  
Esencia simbólica: Cerato  
Cristal simbólico: Apatita"""
    },
    {
        "id": 30,
        "question": "¿Algún día tendré éxito?",
        "answer": """El éxito se vuelve presente cuando tus pasos reflejan tu verdad.  
Esencia simbólica: Impatiens  
Cristal simbólico: Pirita"""
    }
]

# =====================
# 3. EMBEDDINGS
# =====================

texts = [qa["question"] + "\n" + qa["answer"] for qa in qa_pairs]

emb_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
embeddings = [e.embedding for e in emb_response.data]

embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
emb_array = np.array(embeddings).astype("float32")
index.add(emb_array)

def buscar_contexto(pregunta, k=3):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[pregunta]
    ).data[0].embedding

    emb = np.array([emb]).astype("float32")
    distances, indices = index.search(emb, k)

    resultados = []
    for idx in indices[0]:
        resultados.append(qa_pairs[int(idx)])
    return resultados

# =====================
# 4. PROMPT SYSTEM
# =====================

SYSTEM_PROMPT_ES = """
Eres el Oráculo Wise Lucid.

No predices el futuro, no das instrucciones exactas,
no prometes resultados ni haces diagnósticos.

Tono:
- Poético pero claro
- Suave, profundo, meditativo
- Ético, sin supersticiones
- Inspirador, nunca controlador

Puedes mencionar esencias y cristales como símbolos,
no como tratamientos.

Termina con una pregunta suave que invite a la reflexión.
"""

# =====================
# 5. GENERAR RESPUESTA
# =====================

def generar_respuesta(pregunta: str):
    contextos = buscar_contexto(pregunta, k=3)

    contexto_texto = ""
    for c in contextos:
        contexto_texto += f"Pregunta ejemplo: {c['question']}\nRespuesta ejemplo: {c['answer']}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ES},
        {
            "role": "assistant",
            "content": "Estos son ejemplos de cómo sueles responder:\n\n" + contexto_texto
        },
        {
            "role": "user",
            "content": f"Responde en español con tono poético pero claro.\n\nPregunta del usuario:\n{pregunta}"
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.8
    )

    return completion.choices[0].message.content

# =====================
# 6. FASTAPI SERVER
# =====================

app = FastAPI()

# CORS (para Shopify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OracleRequest(BaseModel):
    question: str

class OracleResponse(BaseModel):
    answer: str

@app.post("/api/oracle", response_model=OracleResponse)
def oracle_endpoint(req: OracleRequest):
    ans = generar_respuesta(req.question)
    return OracleResponse(answer=ans)

@app.get("/")
def root():
    return {"message": "Wise Lucid Oracle is running."}
