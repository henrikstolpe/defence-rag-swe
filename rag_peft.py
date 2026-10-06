"""
RAG-aware fine-tuning of AI-Sweden Llama 3 8B Instruct (Swedish version).
Trains the model to:
1. Answer questions grounded in provided Swedish context
2. Refuse to answer when context doesn't contain the information

Single model handles both classification and generation — no model swap needed.
"""
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import Dataset


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open("training_log_rag_llama_swe.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"  # Swedish-trained 8B, single model for classify + generate
OUTPUT_DIR = "./eupolicy_rag_llama_model_swe"
TRAIN_DATA_FILE = "rag_traindata_swe.json"

# ============================================================
# Load training data
# ============================================================
print("Loading Swedish training data...")
with open(TRAIN_DATA_FILE, encoding="utf-8") as f:
    conversations = json.load(f)

print(f"Loaded {len(conversations)} training examples")
dataset = Dataset.from_list(conversations)

# ============================================================
# Load model and tokenizer with 4-bit quantization
# ============================================================
print(f"Loading model: {MODEL_ID}")

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_quant_storage=torch_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map={"": 0},
    torch_dtype=torch_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("Model loaded successfully")

# ============================================================
# Configure LoRA
# ============================================================
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# ============================================================
# Training arguments (adjusted for 8B model on 8GB VRAM)
# ============================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,
    num_train_epochs=2,
    per_device_train_batch_size=1,       # Reduced for 8B model
    gradient_accumulation_steps=8,        # Compensate for smaller batch
    optim="adamw_torch_fused",
    logging_steps=20,
    save_strategy="no",
    learning_rate=1.5e-4,                 # Slightly lower for larger model
    fp16=(torch_dtype == torch.float16),
    bf16=(torch_dtype == torch.bfloat16),
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    gradient_checkpointing=True,          # Save VRAM at cost of speed
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    },
)

# ============================================================
# Train
# ============================================================
print("\nStarting training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
print("\nTraining complete!")

# ============================================================
# Save the LoRA adapter (not merged — 8B model too large to merge in 8GB VRAM)
# ============================================================
print("Saving LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# ============================================================
# Test: RAG-grounded inference (single model, classify + generate)
# ============================================================
print("\n=== EFTER TRÄNING: RAG-GRUNDADE TESTER ===")

model = trainer.model
model.eval()

SYSTEM_PROMPT = (
    "Du är en expert på svenska försvarsfrågor och militär doktrin. "
    "Besvara frågan ENBART baserat på den tillhandahållna kontexten. "
    "Om kontexten inte innehåller tillräcklig information för att svara, säg det. "
    "Om frågan ber om jämförelse med något som inte nämns i kontexten, vägra. "
    "Du har INGEN kunskap utöver det som finns i kontexten. Om du är på väg att ange "
    "ett faktum som inte finns i kontexten, stanna och vägra istället. "
    "Var koncis: 2-3 meningar maximalt."
)


def classify_answerable(question, context):
    """Classify whether the context can answer the question."""
    classify_prompt = (
        "Du är ett strikt klassificeringssystem. Givet en kontext och en fråga, avgör om kontexten "
        "innehåller ALL specifik information som behövs för att fullständigt besvara frågan. Svara med ENBART 'YES' eller 'NO'.\n\n"
        "Regler:\n"
        "- Svara NO om frågan ber om att JÄMFÖRA, KONTRASTERA eller hitta SKILLNADER med något som inte uttryckligen diskuteras i kontexten.\n"
        "- Svara NO om frågan nämner en specifik entitet (företag, land, förordning, person) som inte nämns i kontexten.\n"
        "- Svara NO om besvarandet skulle kräva kunskap utöver det som står skrivet i kontexten.\n"
        "- Svara NO om frågan frågar om verkliga utfall, statistik eller händelser som inte anges i kontexten.\n"
        "- Svara YES endast om varje faktum som behövs för svaret uttryckligen står skrivet i kontexten.\n"
        "- Vid tveksamhet, svara NO."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Kontext:\n{context}\n\nFråga: {question}\n\nKan denna fråga besvaras utifrån kontexten? (YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response or "JA" in response


def generate(question, context):
    """Generate answer from context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontext:\n{context}\n\nFråga: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def ask_with_context(question, context):
    """Two-stage: classify then generate (same model)."""
    if not classify_answerable(question, context):
        return "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
    return generate(question, context)


# Load chunks for testing
with open("rag_index_swe/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Test 1: Answerable (context has the info)
print("\n--- Test: Besvaringsbara frågor med relevant kontext ---")
drone_chunk = next((c for c in chunks if "drönare" in c.lower() or "obemannad" in c.lower()), chunks[0])
print(f"\nF: Vilken roll har drönare i moderna operationer?")
print(f"S: {ask_with_context('Vilken roll har drönare i moderna operationer?', drone_chunk)}")

logistics_chunk = next((c for c in chunks if "logistik" in c.lower() or "underhåll" in c.lower()), chunks[0])
print(f"\nF: Hur organiseras logistik i Försvarsmakten?")
print(f"S: {ask_with_context('Hur organiseras logistik i Försvarsmakten?', logistics_chunk)}")

totaldef_chunk = next((c for c in chunks if "totalförsvar" in c.lower()), chunks[0])
print(f"\nF: Vad är totalförsvaret?")
print(f"S: {ask_with_context('Vad är totalförsvaret?', totaldef_chunk)}")

# Test 2: Unanswerable (context does NOT have the info)
print("\n--- Test: Obesvaringsbara frågor (bör vägra) ---")
irrelevant_chunk = chunks[len(chunks) // 2]

print(f"\nF: Hur mycket kostar en JAS 39 Gripen?")
print(f"S: {ask_with_context('Hur mycket kostar en JAS 39 Gripen?', irrelevant_chunk)}")

print(f"\nF: Hur många soldater har Försvarsmakten totalt?")
print(f"S: {ask_with_context('Hur många soldater har Försvarsmakten totalt?', irrelevant_chunk)}")

comparison_question = "Hur jämför sig svensk drönardoktrin med den ukrainska?"
print(f"\nF: {comparison_question}")
print(f"S: {ask_with_context(comparison_question, drone_chunk)}")

print("\nKlart!")
