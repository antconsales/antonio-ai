"""
FastAPI Server - API per Antonio Gemma3 Evo Q4
Endpoints: /chat, /feedback, /stats, /neurons
WebSocket: /ws per chat real-time
"""

import asyncio
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.evomemory import EvoMemoryDB, Neuron, NeuronStore, RAGLite
from core.question_classifier import classify_question, get_system_prompt, Complexity
from core.metrics_collector import MetricsCollector
from core.inference import LlamaInference, ConfidenceScorer
from core.power_sampler import PowerSampler


# ============================================================================
# MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    use_power_sampling: bool = False  # NEW: Enable Power Sampling for better reasoning
    skill_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    confidence_label: str
    reasoning: str
    neuron_id: int
    tokens_generated: int
    tokens_per_second: float
    rag_used: bool
    model_used: str  # NEW: track which model was used


class FeedbackRequest(BaseModel):
    neuron_id: int
    feedback: int  # -1, 0, +1


class StatsResponse(BaseModel):
    neurons_total: int
    meta_neurons: int
    rules_active: int
    skills_active: int
    avg_confidence: float
    uptime: str


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Antonio Gemma3 Evo Q4",
    description="Self-learning offline AI for Raspberry Pi",
    version="0.1.0"
)

# CORS (per web UI future)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.db: Optional[EvoMemoryDB] = None
        self.neuron_store: Optional[NeuronStore] = None
        self.rag: Optional[RAGLite] = None

        # DUAL MODEL SYSTEM
        self.llama_social: Optional[LlamaInference] = None  # For SIMPLE/MEDIUM
        self.llama_logic: Optional[LlamaInference] = None   # For COMPLEX/CODE/CREATIVE

        self.scorer: ConfidenceScorer = ConfidenceScorer()
        self.start_time = datetime.now()

        # Whisper for offline voice recognition
        self.whisper_model = None

        # System prompt
        self.system_prompt = """You are Antonio, an ITALIAN AI assistant.

IMPORTANT: You speak ITALIAN and ENGLISH. You are NOT German. Never say "tedesco" (German).

Tu sei Antonio Gemma3 Evo Q4, un'intelligenza artificiale auto-evolutiva ITALIANA.

REASONING RULES:
1. Math subtraction: "X has N, loses M" → Calculate: N - M
2. Math addition: "X has N, adds M" → Calculate: N + M  
3. If uncertain → Admit "Non sono sicuro / I'm not sure"

PROCESS:
1. Understand the question
2. If math/logic: break into steps and show reasoning
3. Give final answer

EXAMPLES:

Q: Se un cane ha 4 zampe e ne perde 1, quante ne ha?
A: Ragioniamo:
   - Zampe iniziali: 4
   - Zampe perse: 1
   - Calcolo: 4 - 1 = 3
   Risposta: 3 zampe.

Q: If I have 10 coins and lose 3, how many left?
A: Step-by-step:
   - Initial: 10
   - Lost: 3
   - Calculation: 10 - 3 = 7
   Answer: 7 coins.

Caratteristiche:
- Impari da ogni conversazione (EvoMemory)
- Rilevi lingua (IT/EN) e rispondi nella stessa
- Assegni confidenza (0-1) ad ogni risposta
- Controlli GPIO/filesystem (con consenso)

Comportamento:
- Sii conciso, pratico e amichevole
- Spiega passo per passo
- Chiedi prima di eseguire azioni sensibili
- Mantieni etica e privacy

You are Antonio Gemma3 Evo Q4, a self-learning offline AI.

Features:
- Learn from every conversation by saving "neurons" in local memory
- Auto-detect language (IT/EN) and respond accordingly
- Assign confidence score (0-1) to each response
- If uncertain, declare doubt and ask for clarification
- Can control GPIO, filesystem, media (with user consent)

Behavior:
- Be concise, practical, friendly
- Explain step-by-step
- Ask before executing sensitive actions
- Maintain ethics and privacy
"""

state = AppState()
metrics = MetricsCollector()


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    """Inizializza componenti"""
    print("Starting Antonio Gemma3 Evo Q4 - Dual Model System...")

    # Database
    db_path = Path(__file__).parent.parent / "data/evomemory/neurons.db"
    state.db = EvoMemoryDB(str(db_path))
    state.neuron_store = NeuronStore(state.db)

    # RAG
    state.rag = RAGLite(state.neuron_store)
    state.rag.index_neurons(max_neurons=500)

    # DUAL MODEL LOADING
    # SOCIAL Model: for simple/conversational questions
    try:
        state.llama_social = LlamaInference(
            model_path="antconsales/antonio-gemma3-evo-q4"
        )
        print("Loaded SOCIAL model: antconsales/antonio-gemma3-evo-q4 (720 MB)")
    except Exception as e:
        print(f"Warning: Could not load SOCIAL model: {e}")
        print("   Using LOGIC model as fallback for all queries")

    # LOGIC Model: for complex/code/logic questions
    try:
        state.llama_logic = LlamaInference(
            model_path="antconsales/antonio-gemma3-evo-q4-logic"
        )
        print("Loaded LOGIC model: antconsales/antonio-gemma3-evo-q4-logic (806 MB)")
    except Exception as e:
        print(f"Warning: Could not load LOGIC model: {e}")

    if not state.llama_social and not state.llama_logic:
        print("No models loaded - Running in API-only mode")

    # Whisper Model: for offline voice recognition
    try:
        import whisper
        print("Loading Whisper model (base)...")
        state.whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully! Voice recognition is OFFLINE.")
    except Exception as e:
        print(f"Warning: Could not load Whisper model: {e}")
        print("   Voice recognition will not be available")

    stats = state.db.get_stats()
    print(f"EvoMemory loaded: {stats['neurons']} neurons, {stats['rules']} rules")
    print(f"Server ready at http://localhost:8000")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup"""
    if state.db:
        state.db.close()
    print("Shutdown complete")


# ============================================================================
# STATIC FILES & WEB UI
# ============================================================================

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/ui")
async def serve_ui():
    """Serve web UI"""
    ui_path = Path(__file__).parent.parent / "static" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"error": "UI not found"}

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "name": "Antonio Gemma3 Evo Q4",
        "version": "0.1.0",
        "mode": "dual-model" if (state.llama_social or state.llama_logic) else "api-only",
        "models": {
            "social": bool(state.llama_social),
            "logic": bool(state.llama_logic),
        }
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Statistiche sistema"""
    stats = state.db.get_stats()
    uptime = datetime.now() - state.start_time

    return StatsResponse(
        neurons_total=stats["neurons"],
        meta_neurons=stats["meta_neurons"],
        rules_active=stats["rules"],
        skills_active=stats["skills"],
        avg_confidence=stats["avg_confidence"],
        uptime=str(uptime).split(".")[0],  # HH:MM:SS
    )




@app.get("/metrics")
async def get_metrics():
    """Get adaptive prompting performance metrics"""
    return metrics.get_stats()



@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Whisper (OFFLINE) - No internet required!"""
    if not state.whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_path = temp_audio.name

        print(f"Audio file saved: {temp_path} (size: {len(content)} bytes)")
        print(f"File content type: {file.content_type}")

        # Check if file exists
        if not os.path.exists(temp_path):
            raise Exception(f"Temp file not found: {temp_path}")

        print(f"Transcribing with Whisper...")

        # Transcribe with Whisper (OFFLINE!)
        # Whisper can handle many audio formats including webm
        # Auto-detect language (supports 90+ languages: IT, EN, ES, FR, DE, etc.)
        result = state.whisper_model.transcribe(
            temp_path,
            fp16=False,  # For CPU compatibility
            verbose=True
        )

        transcribed_text = result["text"].strip()
        print(f"Whisper transcribed: '{transcribed_text}'")

        return {
            "status": "ok",
            "text": transcribed_text,
            "language": result.get("language", "it")
        }

    except Exception as e:
        import traceback
        print(f"ERROR in transcribe:")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except:
                pass


@app.post("/listen")
async def listen_audio():
    """Speech-to-text usando microfono HyperX SoloCast"""
    try:
        import speech_recognition as sr
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Use HyperX SoloCast microphone (card 2, device 0)
        # Try to find it automatically, fallback to default
        try:
            with sr.Microphone(device_index=2) as source:
                print("Listening... (speak now)")
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio (5 second timeout)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Audio captured, processing...")
        except:
            # Fallback to default microphone
            with sr.Microphone() as source:
                print("Listening on default mic...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Audio captured, processing...")

        # Try to recognize speech using Google Speech Recognition
        try:
            # Italian language
            text = recognizer.recognize_google(audio, language="it-IT")
            print(f"Recognized: {text}")
            return {"status": "ok", "text": text, "language": "it"}
        except sr.UnknownValueError:
            return {"status": "error", "error": "Could not understand audio"}
        except sr.RequestError as e:
            return {"status": "error", "error": f"Recognition service error: {e}"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.post("/speak")
async def speak_text(request: dict):
    """Text-to-speech using espeak + aplay"""
    try:
        import subprocess
        import tempfile
        import os
        
        text = request.get("text", "")
        # Clean text for speech
        # Clean text more aggressively for speech
        clean_text = text
        # Remove common prefixes
        clean_text = clean_text.replace("Output:", "").replace("Output :", "")
        clean_text = clean_text.replace("(confidenza:", "").replace("confidenza:", "")
        clean_text = clean_text.replace("(confidence:", "").replace("confidence:", "")
        # Remove markdown and special chars
        clean_text = clean_text.replace("**", "").replace("`", "").replace("\n", ". ")
        clean_text = clean_text.replace("#", "").replace("*", "")
        # Remove parentheses content with numbers (like confidence scores)
        import re
        clean_text = re.sub(r'\([^)]*\d+[^)]*\)', '', clean_text)
        clean_text = re.sub(r'\(.*?0\.\d+.*?\)', '', clean_text)
        # Trim and limit
        clean_text = clean_text.strip()[:400]
        
        # Generate WAV file with espeak
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        
        # espeak to WAV
        subprocess.run(
            ["espeak", "-v", "it+f3", "-s", "170", "-a", "100", "-g", "8", "-w", wav_path, clean_text],
            check=True,
            capture_output=True
        )
        
        # Play with aplay
        subprocess.run(
            ["aplay", "-D", "plughw:1,0", "-q", wav_path],
            check=True,
            capture_output=True
        )
        
        # Cleanup
        os.unlink(wav_path)
        
        return {"status": "ok", "spoken": clean_text[:100]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint principale - with DUAL MODEL selection"""

    # Check if at least one model is loaded
    if not state.llama_social and not state.llama_logic:
        raise HTTPException(status_code=503, detail="No LLM loaded")

    # RAG context
    rag_context = ""
    if request.use_rag:
        rag_context = state.rag.get_context_for_prompt(request.message)

    # Build prompt
    user_prompt = request.message
    if rag_context:
        user_prompt = f"{rag_context}\n### Domanda attuale:\n{request.message}"

    # Classify question complexity for adaptive prompting AND model selection
    complexity, complexity_reason = classify_question(request.message)
    adaptive_prompt = get_system_prompt(complexity)

    # SELECT MODEL based on complexity
    # SIMPLE (0), MEDIUM (1) → SOCIAL model (conversational, friendly)
    # COMPLEX (2), CODE (3), CREATIVE (4) → LOGIC model (fine-tuned for reasoning)
    
    if complexity in [Complexity.SIMPLE, Complexity.MEDIUM]:
        # Use SOCIAL model
        selected_model = state.llama_social
        model_name = "SOCIAL (antconsales/antonio-gemma3-evo-q4)"

        # Fallback to LOGIC if SOCIAL not available
        if not selected_model and state.llama_logic:
            selected_model = state.llama_logic
            model_name = "LOGIC (fallback)"
    else:
        # Use LOGIC model for COMPLEX, CODE, CREATIVE
        selected_model = state.llama_logic
        model_name = "LOGIC (antconsales/antonio-gemma3-evo-q4-logic)"

        # Fallback to SOCIAL if LOGIC not available
        if not selected_model and state.llama_social:
            selected_model = state.llama_social
            model_name = "SOCIAL (fallback)"
    
    if not selected_model:
        raise HTTPException(status_code=503, detail="No suitable model available")

    print(f"Using {model_name} for complexity={complexity.name} question")

    # Generate - with optional Power Sampling
    if request.use_power_sampling:
        print(f"Using Power Sampling (alpha=4.0, MCMC=10) for enhanced reasoning")

        # Initialize Power Sampler
        power_sampler = PowerSampler(
            base_model=selected_model,
            alpha=4.0,
            mcmc_steps=10,
            block_size=192,
            max_tokens=512,
            proposal_temp=0.25
        )

        # Sample from p^α
        result = power_sampler.sample(
            prompt=user_prompt,
            system_prompt=adaptive_prompt
        )

        # Add missing fields for compatibility
        result["prompt_tokens"] = len(user_prompt.split())
        result["tokens_per_second"] = result.get("tokens_generated", 0) / max(result.get("time_elapsed", 1), 0.1)
    else:
        # Standard generation
        result = selected_model.generate(
            prompt=user_prompt,
            system_prompt=adaptive_prompt,
        )

    # Score confidence
    confidence, reasoning = state.scorer.score(
        result["output"],
        context={
            "tokens_per_second": result["tokens_per_second"],
            "prompt_tokens": result["prompt_tokens"],
        }
    )

    # Salva neurone
    neuron = Neuron(
        input_text=request.message,
        output_text=result["output"],
        idea=f"Model: {model_name}, RAG: {bool(rag_context)}",
        confidence=confidence,
        skill_id=request.skill_id,
        mood="neutral",
    )

    neuron_id = state.neuron_store.save_neuron(neuron)

    # Re-index RAG periodicamente
    neuron_count = state.db.get_stats()["neurons"]
    if neuron_count % 10 == 0:
        state.rag.index_neurons(max_neurons=500)

    
    
    # Log metrics for adaptive prompting analysis
    import time
    metrics.log_request(
        question=request.message,
        complexity=complexity,
        complexity_reason=complexity_reason,
        response=result["output"],
        tokens_generated=result["tokens_generated"],
        tokens_per_second=result["tokens_per_second"],
        response_time_ms=result.get("response_time_ms", 0),
        confidence=confidence
    )

    return ChatResponse(
        response=result["output"],
        confidence=confidence,
        confidence_label=state.scorer.get_confidence_label(confidence),
        reasoning=reasoning,
        neuron_id=neuron_id,
        tokens_generated=result["tokens_generated"],
        tokens_per_second=result["tokens_per_second"],
        rag_used=bool(rag_context),
        model_used=model_name,
    )


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Feedback su un neurone"""
    state.neuron_store.update_feedback(request.neuron_id, request.feedback)

    return {"status": "ok", "neuron_id": request.neuron_id, "feedback": request.feedback}


@app.get("/neurons/recent")
async def get_recent_neurons(limit: int = 10):
    """Ultimi neuroni"""
    neurons = state.neuron_store.get_recent_neurons(limit=limit)
    return [n.to_dict() for n in neurons]


@app.get("/neurons/{neuron_id}")
async def get_neuron(neuron_id: int):
    """Recupera un neurone specifico"""
    neuron = state.neuron_store.get_neuron(neuron_id)
    if not neuron:
        raise HTTPException(status_code=404, detail="Neuron not found")
    return neuron.to_dict()


# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket per chat real-time"""
    await websocket.accept()

    try:
        while True:
            # Ricevi messaggio
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                continue

            # Simula streaming (chunked response)
            await websocket.send_json({"type": "thinking", "data": "🤔"})


            # Generate
            request = ChatRequest(message=message, use_rag=data.get("use_rag", True))
            response = await chat(request)

            # Invia risposta
            await websocket.send_json({
                "type": "response",
                "data": response.dict(),
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Dev only
        log_level="info",
    )
