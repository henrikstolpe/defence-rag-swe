"""
RAG-aware fine-tuning of Llama 3.2 3B.
Trains the model to:
1. Answer questions grounded in provided context
2. Refuse to answer when context doesn't contain the information
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

log_file = open("training_log_rag_llama.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./eupolicy_rag_llama_model"
TRAIN_DATA_FILE = "rag_traindata.json"

# ============================================================
# Load training data
# ============================================================
print("Loading training data...")
with open(TRAIN_DATA_FILE) as f:
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
# Training arguments
# ============================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch_fused",
    logging_steps=20,
    save_strategy="no",
    learning_rate=2e-4,
    fp16=(torch_dtype == torch.float16),
    bf16=(torch_dtype == torch.bfloat16),
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
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
# Save the merged model
# ============================================================
print("Merging and saving model...")
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ============================================================
# Test: RAG-grounded inference (with two-stage classification)
# ============================================================
print("\n=== AFTER TRAINING: RAG-GROUNDED TESTS ===")

model = merged_model
model.eval()

SYSTEM_PROMPT = (
    "You are an EU AI Act expert. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say so. "
    "If the question asks to compare with something not mentioned in the context, refuse. "
    "You have NO knowledge beyond what is in the context. If you find yourself about to state "
    "a fact not present in the context, stop and refuse instead. "
    "Be concise: 2-3 sentences maximum."
)

def classify_answerable(question, context):
    """Stage 1: Classify whether the context can answer the question."""
    classify_prompt = (
        "You are a strict classification system. Given a context and a question, determine if the context "
        "contains ALL the specific information needed to fully answer the question. Respond with ONLY 'YES' or 'NO'.\n\n"
        "Rules:\n"
        "- Answer NO if the question asks to COMPARE, CONTRAST, or find DIFFERENCES with anything not explicitly discussed in the context.\n"
        "- Answer NO if the question mentions a specific entity (company, country, regulation, person) that is not named in the context.\n"
        "- Answer NO if answering would require knowledge beyond what is written in the context.\n"
        "- Answer NO if the question asks about real-world outcomes, statistics, or events not stated in the context.\n"
        "- Answer YES only if every fact needed for the answer is explicitly written in the context.\n"
        "- When in doubt, answer NO."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nCan this question be answered from the context? (YES/NO)"},
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
    return "YES" in response

def generate(question, context):
    """Stage 2: Generate answer from context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def ask_with_context(question, context):
    """Two-stage: classify then generate."""
    if not classify_answerable(question, context):
        return "The provided context does not contain enough information to answer this question."
    return generate(question, context)

# Load chunks for testing
with open("rag_index/chunks.json") as f:
    chunks = json.load(f)

# Test 1: Answerable (context has the info)
print("\n--- Test: Answerable questions with relevant context ---")
# Find chunk about fines
fines_chunk = next((c for c in chunks if "35 000 000" in c or "7 %" in c), chunks[0])
print(f"\nQ: What are the fines for prohibited AI practices?")
print(f"A: {ask_with_context('What are the fines for prohibited AI practices?', fines_chunk)}")

# Find chunk about social scoring
scoring_chunk = next((c for c in chunks if "social scoring" in c.lower()), chunks[0])
print(f"\nQ: Why is social scoring prohibited?")
print(f"A: {ask_with_context('Why is social scoring prohibited?', scoring_chunk)}")

# Find chunk about sandboxes
sandbox_chunk = next((c for c in chunks if "sandbox" in c.lower()), chunks[0])
print(f"\nQ: What are AI regulatory sandboxes?")
print(f"A: {ask_with_context('What are AI regulatory sandboxes?', sandbox_chunk)}")

# Test 2: Unanswerable (context does NOT have the info)
print("\n--- Test: Unanswerable questions (should refuse) ---")
irrelevant_chunk = next((c for c in chunks if "standardisation" in c.lower() or "harmonised" in c.lower()), chunks[-1])

print(f"\nQ: How many companies have been fined under the AI Act?")
print(f"A: {ask_with_context('How many companies have been fined under the AI Act?', irrelevant_chunk)}")

print(f"\nQ: What is the annual budget of the AI Office?")
print(f"A: {ask_with_context('What is the annual budget of the AI Office?', irrelevant_chunk)}")

china_question = "How does the AI Act compare to China's AI regulations?"
print(f"\nQ: {china_question}")
print(f"A: {ask_with_context(china_question, irrelevant_chunk)}")

print("\nDone!")
