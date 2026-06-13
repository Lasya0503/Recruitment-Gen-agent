import os
import time
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── LiteLLM interceptor: strips cache params Gemini rejects ──────────────────
_original_completion = litellm.completion

def _clean_completion(*args, **kwargs):
    for key in ("cache_prompt", "cache_breakpoint"):
        kwargs.pop(key, None)
    if isinstance(kwargs.get("extra_body"), dict):
        for key in ("cache_prompt", "cache_breakpoint"):
            kwargs["extra_body"].pop(key, None)
        if not kwargs["extra_body"]:
            kwargs.pop("extra_body")
    kwargs["messages"] = [
        {k: v for k, v in m.items() if k not in ("cache_breakpoint", "cache_prompt")}
        if isinstance(m, dict) else m
        for m in kwargs.get("messages", [])
    ]
    
    # 🛡️ RATE LIMIT PROTECTOR: Force a brief 4-second delay before every single API call.
    # This flattens the multi-agent startup spike so you never hit the 15 RPM free-tier limit.
    time.sleep(4)
    
    return _original_completion(*args, **kwargs)

litellm.completion = _clean_completion

# ── Force CrewAI's global default away from its built-in gemini-1.5-flash-8b ─
# CrewAI reads OPENAI_MODEL_NAME and OPENAI_API_KEY as fallbacks internally.
# We point both at Gemini via LiteLLM's OpenAI-compatible prefix.
os.environ.setdefault("OPENAI_MODEL_NAME", "gemini/gemini-2.0-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in your .env file.")

# ── Build LLM config dict — passed explicitly to every Agent ─────────────────
# Using a plain dict instead of crewai.LLM avoids Pydantic validation issues
# across different crewai versions.
def make_llm(temperature: float = 0.2):
    """
    Returns a crewai.LLM instance pointing at gemini-2.0-flash.
    Called fresh for each agent so there's no shared-state issue.
    """
    from crewai import LLM
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        temperature=temperature,
    )

# Pre-built instances used by agents.py
LLM_FAST     = make_llm(temperature=0.2)   # analytical agents
LLM_CREATIVE = make_llm(temperature=0.65)  # email drafting agent

def cooldown_callback(task_output):
    """Plain sleep — safe to call from CrewAI's worker thread."""
    time.sleep(10)