# Agentic Compatibility Annotation Workflow

An agentic workflow for labeling whether **two products are compatible** based on their **text** and optional **image-derived summaries**, with:

- a **main annotation agent**
- a **reflection agent** that critiques the decision
- **retrieval over human-labeled examples**
- **policy memory** that converts failure cases into reusable rules
- **held-out evaluation against human annotators**

This project is designed for product-pair annotation settings such as furniture, home goods, and related retail ecosystems where compatibility depends on factors like:

- room / setting consistency
- functional complementarity
- style coherence
- material consistency
- human labeling patterns
- ambiguity handling

---

## Overview

The system does **not** rely on one single prompt.

Instead, it uses a multi-step workflow:

1. **Feature building**
   - Build structured support signals from text and image summaries.
   - These features are not the final decision.

2. **Retrieval**
   - Retrieve similar **human-labeled product pairs** from memory.
   - Use them as analogies, not hard rules.

3. **Main annotation agent**
   - Predict:
     - `compatible`
     - `incompatible`
     - `uncertain`
   - Return confidence and rationale.

4. **Reflection agent**
   - Check whether the decision is:
     - grounded
     - overconfident
     - inconsistent with retrieved human examples
     - inconsistent with learned policy rules

5. **Policy learning**
   - When the system performs poorly on labeled cases, it converts those mistakes into:
     - **short reusable rules**
     - optionally **a representative example**
   - This keeps the prompt short and policy-oriented.

6. **Evaluation**
   - Compare the final agent output to human annotators on held-out benchmark data.

---

## Main idea

A naive LLM workflow often fails because it:

- over-relies on fluent reasoning
- becomes overconfident on weak evidence
- keeps stuffing more cases into the prompt
- lets prompt length grow too much over time

This project addresses those issues by:

- separating **retrieval memory** from **policy memory**
- learning **short general rules** from failure cases
- only keeping a few **representative examples**
- controlling prompt length with a **budgeted policy slice**

Instead of repeatedly pasting raw wrong cases into the prompt, the system tries to learn compact rules such as:

- Two items should fit the **same setting** to be compatible.
- Cross-room pairs are usually incompatible unless explicitly designed to go together.
- Functional complementarity alone is not enough if the pairing context is implausible.
- Weak evidence should often map to `uncertain`, not a forced hard label.

---

## Project structure

```text
Agentic Workflow/
├── agents.py
├── evaluation.py
├── llm_client.py
├── orchestrator.py
├── policy_memory.py
├── prompts.py
├── retriever.py
├── run_demo.py
├── schemas.py
├── tools.py
└── .gitignore
