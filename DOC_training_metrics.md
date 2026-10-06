# Explanation of Training Metrics (fine-tuning output)

This document explains in detail each field printed during fine-tuning of the
Swedish defence RAG model (`rag_peft.py`). During training, `SFTTrainer`
(TRL/HuggingFace) logs a line like this every 20 steps:

```python
{'loss': '1.573', 'grad_norm': '0.4902', 'learning_rate': '0.0001432',
 'entropy': '1.534', 'num_tokens': '8.192e+04', 'mean_token_accuracy': '0.6707',
 'epoch': '0.3556'}
```

Each line is a snapshot of the model's state at a given *optimization step*. Below,
every field is explained, how to interpret it, and what counts as a good versus a
bad value in this particular setup.

---

## Training setup (context for the numbers)

To make the numbers meaningful, here is the configuration they come from (defined
in `rag_peft.py`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Base model | AI-Sweden Llama 3 8B-instruct | 8 billion parameters |
| Method | QLoRA (4-bit NF4 + LoRA adapter) | Only the adapter is trained |
| Training examples | 450 | Swedish question/context/answer pairs |
| Epochs | 2 | Number of passes over the full dataset |
| `per_device_train_batch_size` | 1 | 1 example per forward pass (8 GB VRAM) |
| `gradient_accumulation_steps` | 8 | Accumulates 8 passes before a weight update |
| **Effective batch size** | **8** | 1 × 8 = 8 examples per optimization step |
| `learning_rate` (start) | 1.5e-4 | Peak value before cosine decay |
| `lr_scheduler_type` | cosine | Learning rate follows a cosine curve |
| `warmup_ratio` | 0.03 | The first ~3 % of steps ramp the LR up |
| `max_grad_norm` | 0.3 | Gradients are clipped at this norm |
| **Total steps** | **114** | 450 examples ÷ 8 ≈ 57 steps/epoch × 2 epochs |

One "step" = one weight update of the LoRA adapter, i.e. after 8 examples have
been processed.

---

## The big picture first (plain language)

Think of fine-tuning as a student studying flashcards. Each flashcard shows some
text, and the student tries to predict the next word. When they get it wrong, they
adjust their understanding a little, then try the next card. After going through
all the cards twice (2 epochs), they have adjusted their understanding thousands
of times.

With that picture in mind, here is what each number is really telling you:

- **loss** → how wrong the student's guesses are (want it to go **down**)
- **mean_token_accuracy** → how often the student guesses exactly right (want it
  **up**)
- **grad_norm** → how big a correction the student makes after each batch of cards
  (want it **steady**, not wild)
- **learning_rate** → how boldly we let the student change their mind (planned to
  shrink over time)
- **entropy** → how confident vs. hesitant the student is when guessing (should
  settle, but not become a know-it-all)
- **num_tokens** → how much material has been studied so far (just a counter)
- **epoch** → how many full passes through the flashcards are done (0 → 2 here)

The rest of this document explains each one in more detail.

---

## Field by field

### `loss` — how wrong the guesses are
**Example value: `1.573`**

This is the most important number. At each point in the text, the model guesses
what the next word (token) should be. Loss measures how far off those guesses are.
A big loss means the model was very surprised by the right answer; a small loss
means it mostly expected the right answer.

- **Lower is better.** 0 would mean perfect prediction (never happens in practice).
- A rough feel: a loss of 1.573 means the model, on average, thought the correct
  next word was only about 21 % likely. As loss drops, that confidence in the
  right answer goes up.
- **What we want to see:** it goes down over time. In this run:
  step 20 = 1.573 → step 40 = 1.143 → step 60 = 0.944. It's learning.

Rough guide for this kind of task:
| Loss | Interpretation |
|------|----------------|
| > 2.5 | Model has barely started learning |
| 1.0–2.0 | Normal early/mid training |
| 0.5–1.0 | Model has adapted well to the domain |
| < 0.3 | Suspiciously low — may be memorizing the data |

### `grad_norm` — how big a correction the model makes
**Example value: `0.4902`**

This is the field people find most confusing, so here is the intuition.

**The analogy:** imagine you're adjusting a shower's hot and cold taps to get the
water temperature right. After each test, you decide how much to turn the taps.
`grad_norm` is basically **how far you decide to turn the taps in one go.**

- The model has many "taps" (its tunable numbers). After looking at a batch of
  examples, it works out, for every tap, *which direction and how much* to turn it
  to make better predictions. That full set of adjustments is called the
  **gradient**.
- `grad_norm` squashes all those individual tap-turns into a single number
  describing the **overall size of the adjustment**. (Mathematically it's the
  length of the gradient vector, but you can just think "total size of this
  update.")

**Why you care about it:**
- If the number is **huge** (say 10, 50, 100), the model is yanking the taps hard
  — like slamming the hot tap fully open. One scalding batch can wreck everything
  learned so far. This is called "exploding gradients," and it makes training blow
  up.
- If the number is **basically zero** for a long time, the model is barely moving
  the taps at all — it has effectively stopped learning ("vanishing gradients").
- If the number is **moderate and steady** (roughly 0.1 to 1.0), the model is
  making sensible, controlled adjustments. That's what you want.

**A built-in safety limit:** this run sets `max_grad_norm = 0.3`. That's a cap: if
the model ever tries to turn the taps harder than 0.3, the system automatically
scales the whole adjustment down so its size is at most 0.3. It stops any single
batch from making a reckless change. (The technical name is "gradient clipping.")

**So why does the log show 0.4902, which is above the 0.3 cap?** Because the number
in the log is measured *before* the cap is applied. It's showing you how big the
raw adjustment wanted to be; the cap then quietly trims it down to 0.3 before it's
actually used. Seeing 0.49 is completely normal and healthy — it just means the
model wanted a slightly bigger step than the cap allows, and the cap did its job.

**Bottom line:** watch for stability. Steady smallish numbers = good. Sudden
spikes into the double or triple digits = trouble. Your 0.49 is fine.

### `learning_rate` — how boldly the model is allowed to change
**Example value: `0.0001432` (another way to write this is 1.432e-4)**

If `grad_norm` is *how big* a correction the model wants to make, `learning_rate`
is a separate dial for *how much of that correction we actually allow* on each
step. Small learning rate = cautious nudges; large = bold changes.

- It is **planned in advance**, not decided by the model. It starts near its peak
  (0.00015) and then slowly shrinks toward almost 0 by the end, following a smooth
  curve (a "cosine" shape).
- **Why shrink it?** Early on you want bold changes so the model learns fast. Near
  the end you want tiny, careful nudges so it settles neatly on a good answer
  instead of overshooting it. Like driving fast on the highway but slowing down to
  park.
- **What we want to see:** a smooth, steady decrease. It's "supposed" to go down —
  that's the plan, not a problem.

### `entropy` — how confident vs. hesitant the model is
**Example value: `1.534`**

When the model guesses the next word, it doesn't pick just one — it spreads its
bet across many possible words. Entropy measures how spread out that bet is.

- **High entropy** = the model is hedging, spreading its guess over lots of words
  ("could be any of these").
- **Low entropy** = the model is confident, putting most of its bet on a few words.
- **What we want to see:** it drifts down slowly as the model grows more sure of
  the material. In this run: step 20 = 1.534 → step 40 = 1.135 → step 60 = 0.980.
- But you don't want it to crash all the way to near-zero. A model that's *too*
  confident becomes a stubborn know-it-all and handles new, unfamiliar questions
  badly. For this system — which must sometimes say "I don't have that in my
  sources" — keeping a little healthy hesitation is actually good.

Quick note: entropy and loss are related but different. Loss asks "was the right
answer among your top bets?" Entropy asks "how spread out were your bets?" — no
matter whether you were right.

### `num_tokens` — how much material has been studied
**Example value: `8.192e+04`, which is 81,920 tokens**

Just a running counter of how many pieces of text (tokens) the model has processed
so far since training began. A token is roughly a word or part of a word.

- It only ever goes **up**. It's not a quality score — it's an odometer.
- It's handy for two things: estimating speed (tokens per second) and confirming
  progress. For example, 81,920 tokens works out to about step 20 in this run,
  which matches when this line was logged.

### `mean_token_accuracy` — how often the model is exactly right
**Example value: `0.6707` (67 %)**

The simplest, most intuitive number. Out of all the next-word guesses in this
batch, what fraction did the model get *exactly* right? Here, 67 %.

- **Higher is better**, and it's always between 0 and 1 (0 % to 100 %).
- **What we want to see:** it climbs over time. In this run: step 20 = 0.671 →
  step 40 = 0.754 → step 60 = 0.787. Clear improvement.
- Rough guide:
  | Accuracy | Interpretation |
  |----------|----------------|
  | < 0.5 | Early training, lots left to learn |
  | 0.6–0.8 | Normal, healthy learning |
  | > 0.9 | Very high — might be memorizing rather than understanding |
- One honest caveat: a lot of text is easy to predict (common words, punctuation,
  formatting), so a high score here doesn't *prove* the answers are good. The real
  test of answer quality is the evaluation suite (the refusal/leakage/retrieval
  tests), not this number.

### `epoch` — how far through the material we are
**Example value: `0.3556`**

How many complete passes through the whole set of flashcards the model has done.

- 0.3556 means it has worked through about 35.6 % of the data once.
- This run does 2 full passes, so the number runs from 0.0 up to 2.0.
- It's your progress bar: when it reaches 2.0, training is done.

---

## How the fields relate

In a healthy training run, the metrics move together:

```
Time passes       →  epoch ↑   num_tokens ↑   learning_rate ↓ (cosine)
Model is learning →  loss ↓    entropy ↓      mean_token_accuracy ↑
Stability         →  grad_norm stays moderate and steady (≈ 0.1–1.0)
```

Comparison between the logged steps in this run:

| Step | epoch | loss | entropy | mean_token_accuracy | learning_rate |
|------|-------|------|---------|---------------------|---------------|
| 20 | 0.3556 | 1.573 | 1.534 | 0.6707 | 1.432e-4 |
| 40 | 0.7111 | 1.143 | 1.135 | 0.7538 | 1.155e-4 |
| 60 | 1.053  | 0.944 | 0.980 | 0.7870 | 7.5e-5 |

Everything points the right way: **loss down, entropy down, accuracy up**, and the
learning rate decaying per the cosine schedule. This is exactly the pattern you
want to see during a healthy fine-tuning run.

---

## What the metrics do NOT tell you

These numbers measure how well the model predicts the *training data*. They say
nothing directly about:

- **Generalization** — whether the model works on new, unseen questions. (There is
  no separate validation loss in this setup; `save_strategy="no"` and no eval
  split.)
- **Refusal vs. leakage** — whether the model correctly refuses to answer when the
  context lacks the information. This is the core of the system and is measured in
  the separate evaluation tests (refusal accuracy, parametric leakage, false
  refusal).
- **Answer quality** — whether the answers are correct, well-structured, and
  properly cited.

That is why the step *after* training — running the extended evaluation tests — is
what decides whether the retrained adapter is actually better than the previous one
(backed up in `adapter_backup_before_retrain/`).

---

## Summary (quick reference)

| Field | Means | Good direction | Bounded to [0,1]? |
|-------|-------|----------------|--------------------|
| `loss` | Cross-entropy on next token | ↓ lower | No (0 → ∞) |
| `grad_norm` | Size of gradient step | stable, moderate | No |
| `learning_rate` | Step size (scheduled) | ↓ cosine decay | No |
| `entropy` | Uncertainty of the distribution | ↓ slowly (not to 0) | No |
| `num_tokens` | Cumulative token count | ↑ (bookkeeping) | No |
| `mean_token_accuracy` | Fraction of tokens guessed correctly | ↑ higher (but not → 1.0) | Yes |
| `epoch` | Progress through the data | ↑ to 2.0 | No (0 → 2) |
