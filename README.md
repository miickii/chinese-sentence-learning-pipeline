# Chinese Sentence Curriculum Generator

This project builds a **self-updating, Chinese-first sentence generation system for language learning**.

The system generates high-quality Chinese sentences, verifies them using an ensemble of open-source language models, and accepts only sentences that are pedagogically useful:
- exactly **one new Chinese vocabulary item**
- at most **one novel grammar pattern** (discovered statistically)
- low similarity to previously accepted sentences

After a one-time bootstrap from an existing Anki deck, the pipeline becomes the **single source of truth** and continuously updates its internal model of what the learner knows.

English is generated **only after acceptance** as a gloss for display and never influences difficulty or progression.

---

## Motivation

Most language-learning systems suffer from one of two problems:
1. Static curricula that do not adapt to what the learner actually knows
2. Uncontrolled AI generation that introduces too much novelty at once

This project addresses both by enforcing **controlled novelty**:
> Every new sentence is mostly familiar, but introduces exactly one new thing.

Grammar is not predefined or labeled. Instead, it is **discovered statistically** from real usage and tracked dynamically as the system runs.

---

## Core Design Principles

- **Chinese-first**: difficulty and novelty are computed only from Chinese
- **No preset grammar lists**: grammar emerges from patterns in the data
- **Rejection sampling**: models propose freely; the pipeline decides
- **Ensemble verification**: quality is enforced by multiple models
- **Dynamic state**: vocabulary and grammar familiarity update after every sentence

---

## What the System Does

At a high level, the system runs a loop:

1. Generate many candidate Chinese sentences locally
2. Verify quality using an ensemble of judge models
3. Filter candidates deterministically:
   - Chinese vocabulary novelty
   - grammar pattern novelty (Layer B + C)
   - similarity to past sentences
4. Accept the best candidate
5. Update internal state (vocab + grammar)
6. Optionally generate an English gloss
7. Repeat

This loop never stagnates and never requires manual grammar supervision.

---

## Project Structure

