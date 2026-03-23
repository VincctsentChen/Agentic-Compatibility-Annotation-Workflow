# Agentic Compatibility Annotation Workflow

An agentic workflow for labeling whether **two products are compatible** based on their **text** and optional **image-derived summaries**. To run the demo, type "python run_demo.py". The project has following subsections:

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

## File descriptions

### `schemas.py`
Defines the main data structures used throughout the project, such as:

- `Product`
- `ProductPair`
- `RetrievedExample`
- `SupportEvidence`
- `AnnotationDecision`
- `ReflectionFeedback`
- `LearnedRule`
- `RepresentativeExample`
- `PolicySlice`
- `PipelineOutput`

### `tools.py`
Builds deterministic support evidence from the current product pair.

Typical outputs include:

- compact product text
- token overlap
- shared keywords
- category information
- image summary presence
- other lightweight support signals

These features are **inputs** to the agents, not the final compatibility decision.

### `retriever.py`
Stores and retrieves similar **human-labeled** training pairs.

This retrieval layer is used for:

- few-shot grounding
- checking how humans labeled similar situations
- exposing ambiguity in similar examples

### `prompts.py`
Builds compact prompts for:

- the main annotator
- the reflection agent
- the policy learner

This file is important because it controls prompt size by:

- compressing product information
- using only a few retrieved examples
- inserting learned rules instead of long raw case dumps

### `agents.py`
Contains the main agent logic:

- `MainAnnotatorAgent`
- `ReflectionAgent`
- `PolicyLearnerAgent`

Responsibilities:

- call the LLM
- parse structured JSON outputs
- reflect on earlier decisions
- convert wrong cases into reusable policy rules

### `policy_memory.py`
Stores and serves learned policy information.

This memory contains:

- short reusable rules
- optional representative examples

It also enforces a prompt budget so the prompt does not grow without bound.

### `orchestrator.py`
Coordinates the full workflow:

- build support evidence
- retrieve examples
- get a policy slice
- run the main annotator
- run the reflector
- retry up to `max_turn`
- evaluate against human labels
- learn new policy rules when useful

### `evaluation.py`
Evaluates the agent against human annotations using metrics such as:

- majority vote accuracy
- average agent-human agreement
- human consensus strength
- uncertain rate
- uncertain rate under human disagreement

### `llm_client.py`
Provides the model client wrapper.

You can connect it to:

- a mock client for development
- DashScope / Qwen for real inference

### `run_demo.py`
Demo entry point.

This file typically:

- creates training memory
- creates benchmark pairs
- instantiates the workflow
- runs annotation and evaluation
- prints turn-by-turn traces

---

## Workflow diagram

```text
Raw pair (text + optional image summaries)
        |
        v
Support evidence builder
        |
        v
Retrieve similar human-labeled pairs
        |
        v
Policy memory slice
(short rules + representative examples)
        |
        v
Main annotation agent
(label + confidence + rationale)
        |
        v
Reflection agent
(check grounding / confidence / policy consistency)
        |
   +----+----+
   |         |
 accept     revise prompt
   |         |
   |      retry (max_turn)
   v
Final prediction
        |
        v
Compare to held-out human labels
        |
        v
Policy learner updates reusable rules
