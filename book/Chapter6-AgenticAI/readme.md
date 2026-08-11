# Chapter 6: Agentic AI for Science

In 2023, language models autocompleted code. In 2026, agentic systems read whole repositories, run code and inspect the results, search the literature, and carry out multi-step research tasks with limited supervision. Between those two dates, the useful human skill shifted. It is no longer prompting — asking nicely, in the right words, is a solved problem. The skills this chapter teaches are **specification** (stating a task precisely enough that success is checkable) and **evaluation** (checking it).

Nothing here contradicts Chapters 1 through 5; it depends on them. An agent's output is an untrusted analysis, and you already know what to do with an untrusted analysis: demand a reproducible environment (Chapter 5), a fair evaluation against a held-out truth (Chapter 3), and a baseline it must beat. This chapter applies that machinery to the agents themselves.

A note on how we work in this chapter: the notebooks do not call any external AI service. They run against provided artifacts — recorded agent answers, simulated agents with planted failure modes — so they execute in CI and cost nothing. The reasoning transfers directly to the live agents you use in your project, where the course AI policy applies: use is allowed, disclosed, and verified, and you must be able to defend every line you submit ([Chapter 1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md)).

## What is in this chapter

1. **[6.1 From language models to agents](6.1_llms_to_agents.md)** — Concepts: what a transformer-based LLM is and is not, why LLMs are bad calculators, retrieval, tool use, and hallucination as a failure mode distinct from ordinary error.
2. **[6.2 Critical evaluation of AI output](6.2_critical_evaluation.ipynb)** — Hands-on: verify an agent's quantitative claims against data, catch a fabricated citation, and score two AI reviews to see judge biases in action.
3. **[6.3 Build an evaluation set](6.3_build_an_eval_set.ipynb)** — The core graded exercise of the chapter: before trusting an agent with a task, build the eval set that measures it. Worked end to end on a GNSS velocity task, then repeated on a task from your own project domain.
4. **[6.4 Disclosure and norms](6.4_disclosure_and_norms.md)** — Attribution and disclosure for AI-assisted research: what journals expect, the course disclosure format, and who owns correctness (you).

## Learning outcomes

By the end of this chapter you can:

- explain what an agent is (an LLM, a set of tools, and a loop over observations) and where in that loop errors enter;
- verify an AI-generated claim against the underlying data rather than against your impression of plausibility;
- design and run an evaluation set with ground truth for an agent task before deploying the agent;
- disclose AI assistance in the format the course and current journal policies require.
