# Voice Intelligence Module — Design Document

## Purpose

Transcribe recorded speech to text and inject it as structured intelligence into the UND module (positions, unit activities, vehicle counts, etc.).

## Architecture

```
Microphone/Audio file
       │
       ▼
┌─────────────────────────┐
│  Speech-to-Text (STT)   │
│  Whisper (local) or     │
│  cloud API              │
└─────────────────────────┘
       │  Raw Swedish text
       ▼
┌─────────────────────────┐
│  NER/Parser (LLM)       │
│  Extract structured     │
│  intel from free text   │
└─────────────────────────┘
       │  entity_id, attribute, value, confidence
       ▼
┌─────────────────────────┐
│  UND.inject()           │
│  Update simulation      │
│  state                  │
└─────────────────────────┘
```

## Component 1: Speech-to-Text

Two options depending on needs:

| Option | Model | Runs on | Swedish quality | Latency |
|--------|-------|---------|-----------------|---------|
| Local | `openai/whisper-large-v3` | Your GPU (4GB) | Excellent | ~5s per 30s audio |
| Cloud | Whisper API or Google STT | Remote | Excellent | ~2s |

Since the GPU is 8GB and the LLM takes ~5GB, options are:
- Unload the LLM during transcription, or
- Use `whisper-medium` (~1.5GB) which fits alongside the LLM, or
- Use the Whisper API (offload to cloud)

## Component 2: Text → Structured Intelligence (the hard part)

Raw transcription might look like:
> "Kustjägarna rapporterar fyra BMD vid vägkorsningen utanför Mölnlycke, riktning väst, uppskattad hastighet femton kilometer i timmen"

This needs to be parsed into:
```python
und.inject("VDV_MAIN", "position", [57.670, 12.10], source="KJ rapport (radio)")
und.inject("VDV_MAIN", "vehicles", {"BMD-4M": 4}, source="KJ rapport (radio)")
und.inject("VDV_MAIN", "heading", 270, source="KJ rapport (radio)")
und.inject("VDV_MAIN", "speed_kmh", 15.0, source="KJ rapport (radio)")
```

Best approach: send the transcribed text to Claude with a structured output prompt that maps free-text reports to UND entity IDs and attributes. The prompt would include the current UND entity list so the LLM knows what IDs are valid.

## Component 3: Integration with UND

Straightforward — just call `und.inject()` with the parsed output.

## Implementation Sketch

```python
# voice_intel.py — Speech-to-Intelligence pipeline

import whisper  # or openai-whisper
from und import UND

class VoiceIntel:
    def __init__(self, und_instance: UND):
        self.und = und_instance
        self.model = whisper.load_model("medium")  # fits in remaining VRAM

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to Swedish text."""
        result = self.model.transcribe(audio_path, language="sv")
        return result["text"]

    def parse_intel(self, text: str) -> list[dict]:
        """Use Claude to extract structured intel from free text."""
        # Send text + UND entity list to Claude
        # Return list of {entity_id, attribute, value, confidence}
        ...

    def process_audio(self, audio_path: str):
        """Full pipeline: audio → text → structured intel → UND injection."""
        text = self.transcribe(audio_path)
        intel_items = self.parse_intel(text)
        for item in intel_items:
            self.und.inject(**item, source="Voice report (radio)")
```

## Key Decisions Needed

1. **Real-time or batch?** — Continuous listening vs. recording clips to process
2. **Whisper model size** — `medium` (1.5GB) fits alongside LLM, `large-v3` (3GB) needs model swap
3. **Parser** — Claude API (best quality, costs money) vs. local 8B model (free, less reliable for structured extraction)
4. **Location resolution** — How to map place names ("Mölnlycke", "vid bron") to coordinates. Needs a gazetteer/geocoder for the operational area.
5. **Confidence** — Voice reports could default to 0.7-0.8 confidence, with the LLM parser assessing ambiguity

## Dependencies

```bash
pip install openai-whisper  # or faster-whisper for optimized inference
```

## VRAM Budget

```
LLM (4-bit, inference): ~5.0 GB
Whisper medium:         ~1.5 GB
Free:                   ~1.1 GB
─────────────────────────────
Total:                   7.6 GB (fits RTX 4060 8GB)
```

If using whisper-large-v3 (3GB), the LLM must be unloaded first.
