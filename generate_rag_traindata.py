"""
Generate RAG-aware training data for the Llama 3.2 model.
Creates two types of examples:
1. ANSWERABLE: Context contains the answer -> model extracts it
2. UNANSWERABLE: Context does NOT contain the answer -> model refuses

Tailored for the EU AI Act (Regulation 2024/1689).
"""
import json
import random

random.seed(42)

# Load the document chunks (same ones used by the RAG index)
with open("rag_index/chunks.json") as f:
    chunks = json.load(f)

# ============================================================
# ANSWERABLE examples: question + relevant context -> answer
# ============================================================
answerable = [
    {
        "context_keywords": ["35 000 000", "7 %", "Article 5", "prohibited"],
        "questions": [
            "What are the fines for violating prohibited AI practices under the EU AI Act?",
            "What penalties apply for non-compliance with Article 5 of the AI Act?",
            "How much can companies be fined for prohibited AI practices?",
        ],
        "answer": "Non-compliance with the prohibited AI practices in Article 5 is subject to administrative fines of up to 35 million euros or, if the offender is an undertaking, up to 7 percent of its total worldwide annual turnover for the preceding financial year, whichever is higher."
    },
    {
        "context_keywords": ["15 000 000", "3 %", "operators", "notified bodies"],
        "questions": [
            "What fines apply for non-compliance with high-risk AI system obligations?",
            "What are the penalties for operators who violate AI Act requirements?",
            "How much can providers be fined for failing to meet high-risk AI obligations?",
        ],
        "answer": "Non-compliance with obligations related to operators or notified bodies, other than the prohibited practices, is subject to administrative fines of up to 15 million euros or up to 3 percent of total worldwide annual turnover for the preceding financial year, whichever is higher."
    },
    {
        "context_keywords": ["7 500 000", "1 %", "incorrect", "misleading information"],
        "questions": [
            "What is the fine for providing misleading information to authorities under the AI Act?",
            "What penalty applies for supplying incorrect information to notified bodies?",
            "How much can companies be fined for giving false information under the EU AI Act?",
        ],
        "answer": "The supply of incorrect, incomplete or misleading information to notified bodies or national competent authorities is subject to administrative fines of up to 7.5 million euros or up to 1 percent of total worldwide annual turnover for the preceding financial year, whichever is higher."
    },
    {
        "context_keywords": ["social scoring", "discriminatory", "dignity", "evaluate or classify"],
        "questions": [
            "Why is social scoring by AI systems prohibited under the EU AI Act?",
            "What does the AI Act say about AI-based social scoring?",
            "Are social scoring AI systems allowed in the EU?",
        ],
        "answer": "AI systems providing social scoring of natural persons by public or private actors are prohibited because they may lead to discriminatory outcomes and exclusion of certain groups, violating the right to dignity and non-discrimination and the values of equality and justice."
    },
    {
        "context_keywords": ["subliminal", "manipulative", "distorting human behaviour", "autonomy"],
        "questions": [
            "What manipulative AI practices are prohibited under the EU AI Act?",
            "Does the AI Act ban subliminal manipulation by AI systems?",
            "What does the EU AI Act say about AI systems that manipulate behaviour?",
        ],
        "answer": "The AI Act prohibits AI systems that deploy subliminal techniques beyond human perception or other manipulative or deceptive techniques that subvert or impair a person's autonomy, decision-making or free choice, particularly when they are likely to cause significant physical, psychological or financial harm."
    },
    {
        "context_keywords": ["biometric identification", "real-time", "publicly accessible", "law enforcement"],
        "questions": [
            "Is real-time biometric identification allowed in public spaces under the AI Act?",
            "What are the rules on facial recognition in public spaces?",
            "When can law enforcement use real-time remote biometric identification?",
        ],
        "answer": "Real-time remote biometric identification in publicly accessible spaces for law enforcement is generally prohibited, except in narrowly defined situations involving the search for victims of crime, imminent threats to life or physical safety, or the identification of suspects of serious criminal offences."
    },
    {
        "context_keywords": ["emotion recognition", "workplace", "education", "prohibited"],
        "questions": [
            "Are emotion recognition AI systems allowed in workplaces under the EU AI Act?",
            "What does the AI Act say about emotion detection in education?",
            "Is it legal to use AI to detect emotions of employees or students?",
        ],
        "answer": "The EU AI Act prohibits AI systems intended to detect the emotional state of individuals in workplace and education situations. This prohibition does not cover AI systems placed on the market strictly for medical or safety reasons, such as systems intended for therapeutical use."
    },
    {
        "context_keywords": ["high-risk", "education", "vocational training", "evaluating learning"],
        "questions": [
            "Are AI systems used in education classified as high-risk?",
            "What educational AI systems are considered high-risk under the AI Act?",
            "Does the AI Act regulate AI in schools and universities?",
        ],
        "answer": "AI systems used in education or vocational training are classified as high-risk when used for determining access or admission, assigning persons to programmes, evaluating learning outcomes, assessing appropriate education levels, or monitoring and detecting prohibited behaviour of students during tests."
    },
    {
        "context_keywords": ["risk-based approach", "prohibit", "high-risk", "transparency obligations"],
        "questions": [
            "What approach does the EU AI Act take to regulating AI?",
            "How does the AI Act categorize AI systems by risk?",
            "What is the risk-based approach in the EU AI Act?",
        ],
        "answer": "The EU AI Act follows a risk-based approach that tailors rules to the intensity and scope of risks that AI systems can generate. It prohibits certain unacceptable AI practices, lays down mandatory requirements for high-risk AI systems, and establishes transparency obligations for certain other AI systems."
    },
    {
        "context_keywords": ["general-purpose AI", "systemic risk", "floating point operations", "threshold"],
        "questions": [
            "What is a general-purpose AI model with systemic risk?",
            "How does the AI Act define systemic risk for AI models?",
            "What threshold determines if a general-purpose AI model has systemic risk?",
        ],
        "answer": "A general-purpose AI model is presumed to have systemic risk if the cumulative amount of computation used for its training, measured in floating point operations, meets a set threshold. This threshold is to be adjusted over time to reflect technological and industrial changes such as algorithmic improvements or increased hardware efficiency."
    },
    {
        "context_keywords": ["AI Office", "codes of practice", "facilitate", "drawing up"],
        "questions": [
            "What is the role of the AI Office under the EU AI Act?",
            "What does the AI Office do regarding codes of practice?",
            "How does the AI Office support compliance with the AI Act?",
        ],
        "answer": "The AI Office encourages and facilitates the drawing up, review and adaptation of codes of practice, collaborating with relevant national competent authorities, civil society organisations, stakeholders and experts including the Scientific Panel. It also monitors general-purpose AI models for systemic risks."
    },
    {
        "context_keywords": ["sandbox", "innovation", "testing", "development"],
        "questions": [
            "What are AI regulatory sandboxes under the EU AI Act?",
            "How does the AI Act support innovation through sandboxes?",
            "What is the purpose of regulatory sandboxes in the AI Act?",
        ],
        "answer": "AI regulatory sandboxes are controlled experimentation and testing environments established by Member States to facilitate the development and testing of innovative AI systems under strict regulatory oversight before they are placed on the market. They aim to foster AI innovation while ensuring compliance with the Regulation."
    },
    {
        "context_keywords": ["deployer", "natural or legal person", "authority", "using an AI system"],
        "questions": [
            "What is a deployer under the EU AI Act?",
            "How does the AI Act define who deploys an AI system?",
            "Who counts as a deployer in the context of the AI Act?",
        ],
        "answer": "A deployer is any natural or legal person, including a public authority, agency or other body, using an AI system under its authority, except where the AI system is used in the course of a personal non-professional activity."
    },
    {
        "context_keywords": ["CE marking", "conformity", "internal market", "free movement"],
        "questions": [
            "Do high-risk AI systems need CE marking under the EU AI Act?",
            "What is the role of CE marking for AI systems?",
            "How do AI systems demonstrate conformity with the AI Act?",
        ],
        "answer": "High-risk AI systems must bear the CE marking to indicate their conformity with the Regulation so they can move freely within the internal market. For AI systems embedded in a product, a physical CE marking should be affixed and may be complemented by a digital CE marking."
    },
    {
        "context_keywords": ["military", "defence", "national security", "excluded"],
        "questions": [
            "Does the EU AI Act apply to military AI systems?",
            "Are defence and national security AI systems covered by the AI Act?",
            "What AI systems are excluded from the scope of the EU AI Act?",
        ],
        "answer": "AI systems placed on the market, put into service, or used for military, defence or national security purposes are excluded from the scope of the AI Act. However, if such a system is used outside those purposes temporarily or permanently for civilian or law enforcement purposes, it falls within the scope of the Regulation."
    },
]

# ============================================================
# UNANSWERABLE examples: question + irrelevant context -> refusal
# ============================================================
unanswerable_questions = [
    "How many AI systems have been registered in the EU database so far?",
    "What is the annual budget of the AI Office?",
    "Which companies have been fined under the EU AI Act?",
    "How many regulatory sandboxes are currently operating in the EU?",
    "What is the market size of the EU AI industry?",
    "How many employees work at the AI Office?",
    "What specific AI systems has OpenAI registered under the AI Act?",
    "What is the EU's total spending on AI research and development?",
    "How many high-risk AI systems are currently deployed in the EU?",
    "What is the timeline for the EU AI Act review in 2030?",
    "Which Member State was the first to implement the AI Act nationally?",
    "How does the AI Act compare to China's AI regulations?",
    "What is the average cost of AI Act compliance for companies?",
    "How many notified bodies have been designated under the AI Act?",
    "What percentage of EU companies use high-risk AI systems?",
    # Adversarial: topics the model has strong parametric knowledge about
    "How does the EU AI Act differ from the US Executive Order on AI?",
    "What did OpenAI say about the EU AI Act?",
    "How does GDPR interact with the AI Act in practice?",
    "What is the role of ChatGPT under the EU AI Act?",
    "How many parameters does GPT-4 have and is it classified as high-risk?",
    "What is Google DeepMind's compliance strategy for the AI Act?",
    "How does the AI Act regulate autonomous vehicles specifically?",
    "What happened at the AI Safety Summit regarding the EU AI Act?",
    "How much did lobbying against the AI Act cost tech companies?",
    "What is Meta's position on open-source AI under the AI Act?",
    "How does the AI Act apply to GitHub Copilot?",
    "What fines has the EU issued for AI violations so far?",
    "How does the UK AI framework compare to the EU AI Act?",
    "What is Microsoft's compliance plan for the EU AI Act?",
    "How many jobs will the AI Act create or destroy in Europe?",
]

# Adversarial comparison questions: context IS about the AI Act, but the question
# asks to compare with something external that isn't in the context.
# These teach the model that having context about ONE side of a comparison
# is not enough — it must refuse if the other side isn't in the context.
comparison_unanswerable = [
    "How does the EU AI Act compare to China's AI governance framework?",
    "What are the differences between the EU AI Act and the US approach to AI regulation?",
    "How does the AI Act's risk-based approach differ from the UK's pro-innovation framework?",
    "What are the similarities between the EU AI Act and Canada's AIDA?",
    "How does the AI Act compare to Japan's AI governance guidelines?",
    "What are the key differences between the EU AI Act and Brazil's AI regulation?",
    "How does the EU's approach to AI differ from Singapore's Model AI Governance Framework?",
    "How does the AI Act compare to South Korea's AI legislation?",
    "What are the differences between the EU AI Act and India's approach to AI regulation?",
    "How does the AI Act's definition of AI compare to the OECD definition?",
    "How do the AI Act's penalties compare to GDPR fines in practice?",
    "What are the differences between the AI Act and the Digital Services Act regarding AI?",
    "How does the EU AI Act compare to ISO 42001 on AI management systems?",
    "How does the AI Act's transparency requirement compare to the US AI Bill of Rights?",
    "What are the differences between the AI Act and Australia's AI Ethics Framework?",
]

refusal_templates = [
    "The provided context does not contain specific information to answer this question.",
    "Based on the available context, I cannot find information addressing this question.",
    "This question cannot be answered from the provided context as it does not contain the relevant details.",
    "The context does not include information about this topic.",
    "I don't have enough information in the provided context to answer this accurately.",
]

# ============================================================
# Find relevant chunks for each answerable example
# ============================================================
def find_relevant_chunk(keywords):
    """Find the chunk that best matches the keywords."""
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

def find_irrelevant_chunk(exclude_keywords):
    """Find a chunk that does NOT match the keywords."""
    candidates = []
    for chunk in chunks:
        score = sum(1 for kw in exclude_keywords if kw.lower() in chunk.lower())
        if score == 0:
            candidates.append(chunk)
    if candidates:
        return random.choice(candidates)
    return random.choice(chunks)  # fallback

# ============================================================
# Generate training examples (multi-chunk format matching inference)
# ============================================================
training_examples = []

SYSTEM_PROMPT = (
    "You are an EU AI Act expert. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say so. "
    "If the question asks to compare with something not mentioned in the context, refuse. "
    "You have NO knowledge beyond what is in the context. If you find yourself about to state "
    "a fact not present in the context, stop and refuse instead. "
    "Be concise: 2-3 sentences maximum."
)

def build_multi_chunk_context(primary_chunk, n_extra=2):
    """Build a multi-chunk context matching inference format: 1 relevant + N random chunks."""
    sources = []
    # Primary relevant chunk gets high score
    sources.append(f"[Source 1, relevance: {random.uniform(0.70, 0.95):.2f}]: {primary_chunk}")
    # Add extra chunks (less relevant) to simulate real retrieval
    used = {primary_chunk}
    for i in range(n_extra):
        extra = random.choice(chunks)
        while extra in used:
            extra = random.choice(chunks)
        used.add(extra)
        score = random.uniform(0.30, 0.60)
        sources.append(f"[Source {i+2}, relevance: {score:.2f}]: {extra}")
    return "\n\n".join(sources)

def build_irrelevant_multi_chunk_context(n_chunks=3):
    """Build multi-chunk context where none are relevant."""
    sources = []
    used = set()
    for i in range(n_chunks):
        chunk = random.choice(chunks)
        while chunk in used:
            chunk = random.choice(chunks)
        used.add(chunk)
        score = random.uniform(0.30, 0.55)
        sources.append(f"[Source {i+1}, relevance: {score:.2f}]: {chunk}")
    return "\n\n".join(sources)

# Answerable examples (multi-chunk format)
for item in answerable:
    context = find_relevant_chunk(item["context_keywords"])
    if not context:
        continue
    for question in item["questions"]:
        multi_context = build_multi_chunk_context(context)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{multi_context}\n\nQuestion: {question}"},
                {"role": "assistant", "content": item["answer"]},
            ]
        })

# Unanswerable examples (multi-chunk format, irrelevant context)
for question in unanswerable_questions:
    q_words = question.lower().split()
    for _ in range(3):
        multi_context = build_irrelevant_multi_chunk_context()
        refusal = random.choice(refusal_templates)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{multi_context}\n\nQuestion: {question}"},
                {"role": "assistant", "content": refusal},
            ]
        })

# Comparison unanswerable examples: RELEVANT AI Act chunks but still refuse
for question in comparison_unanswerable:
    for _ in range(3):
        # Use a relevant chunk as primary but question asks about external topic
        primary = random.choice(chunks)
        multi_context = build_multi_chunk_context(primary)
        refusal = random.choice(refusal_templates)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{multi_context}\n\nQuestion: {question}"},
                {"role": "assistant", "content": refusal},
            ]
        })

# Shuffle and repeat for reinforcement
random.shuffle(training_examples)
all_examples = []
for _ in range(4):
    shuffled = list(training_examples)
    random.shuffle(shuffled)
    all_examples.extend(shuffled)

# Save
with open("rag_traindata.json", "w") as f:
    json.dump(all_examples, f, indent=2)

n_answerable = sum(1 for ex in training_examples if not ex["messages"][2]["content"].startswith("The provided") and not ex["messages"][2]["content"].startswith("Based on") and not ex["messages"][2]["content"].startswith("This question") and not ex["messages"][2]["content"].startswith("The context") and not ex["messages"][2]["content"].startswith("I don't"))
n_unanswerable = len(training_examples) - n_answerable

print(f"Generated {len(all_examples)} total training examples")
print(f"  Answerable: {n_answerable} unique ({n_answerable*4} with repeats)")
print(f"  Unanswerable: {n_unanswerable} unique ({n_unanswerable*4} with repeats)")
print(f"Saved to rag_traindata.json")
