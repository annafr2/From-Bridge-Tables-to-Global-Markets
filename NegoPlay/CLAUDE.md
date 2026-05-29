# CLAUDE.md

> Instructions for AI coding assistants (Antigravity's Gemini 3 Pro agent,
> Claude, Cursor, etc.) working on NegoPlay.
> **Read this file at the start of every session.**

---

## 👤 About the Developer

**Anna Ben-Shushan** — Lecturer at Sami Shamoon College of Engineering,
PhD candidate at LUT University (Finland), AI consultant.

- **Primary language for communication:** Hebrew (עברית)
- **Code language:** English (English variable names, English comments)
- **Documentation:** English (for international academic + dev audience)
- **Code style preference:** Clean, professional, production-grade
- **Pedagogical approach:** Prefers full copy-pasteable code blocks,
  not diff-style edits

---

## 🎯 Project Overview (Read First)

**NegoPlay** investigates whether decision-making profiles from elite bridge
players (149K tournament hands) can predict behavior in business negotiations
via LLM agents.

**Research question:**
> *Can LLM agents built from bridge profiles exhibit ≥70% behavioral
> alignment between winning in bridge and winning in negotiation
> simulations?*

This is a **course final project** (AI Development Expert, 5 weeks) AND
the **empirical baseline** for Paper 1 of Anna's PhD.

**Important:** Bridge is the *laboratory*. The research target is *business
negotiation*. Frame discussions accordingly.

---

## 🏗️ Technical Stack

### Required
- **Python 3.11+** with `venv`
- **WSL2 on Windows** as development OS
- **Google Antigravity** as primary IDE (agent-first development)
- **Google Gemini API** as the *default / preferred* LLM provider for cost
  reasons — but **Anthropic Claude** and **OpenAI** keys are also available
  and may be used when justified.

### LLM provider policy

**Default: Gemini Flash 2.0.** Cost efficiency — the whole NegoPlay MVP
should run under $10 in Gemini calls. Use Gemini unless there's a concrete
reason to switch.

**When it's OK to use Claude or OpenAI:**
- A specific task needs a capability where Gemini measurably underperforms
  (e.g. long-context reasoning, tool use, structured output reliability)
  and the difference matters for the experiment.
- Cross-model validation: running the same prompt on 2–3 providers to show
  that the finding is not provider-specific (good for the PhD paper).
- Final synthesis / paper-writing assistance, where quality > cost.

**Rules whichever provider is used:**
- All keys go through `.env` (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`). Never hardcode.
- Route everything through `src/shared/llm_client.py` so the call site
  doesn't care which provider answered. The client picks the provider from
  a `provider=` argument that defaults to `"gemini"`.
- Log provider + model + token counts + cost for every call to
  `results/llm_logs/` — including cross-provider comparisons.
- Budget alert at $20 cumulative across **all** providers; hard cap $50.

**Note:** Antigravity's coding agent uses Gemini 3 Pro internally (different
from project's Gemini Flash 2.0 calls). These are separate billing/usage.

### Libraries (approved)
```python
# Data & ML
pandas
numpy
scikit-learn   # K-Means, HDBSCAN, KNN, metrics, StandardScaler
gensim         # Word2Vec
shap           # XAI for cluster explanation

# LLM providers (Gemini = default; others available when justified)
google-generativeai  # Gemini Flash 2.0 / Pro — default
anthropic            # Claude — for cross-validation or quality tasks
openai               # OpenAI — for cross-validation or quality tasks

# Utilities
python-dotenv
pydantic       # Type validation
tqdm           # Progress bars

# Testing
pytest
pytest-cov

# Optional / Stretch
torch          # If LSTM stretch goal is pursued
```

### Forbidden / Out of Scope
- ❌ TensorFlow (use PyTorch if any DL)
- ❌ Fine-tuning LLMs (out of MVP scope)
- ❌ Diffusion models (irrelevant)
- ❌ External databases (CSV files are sufficient)
- ❌ Web frameworks (this is a research project, not a web app)

---

## 🔧 Slash Commands (Claude Code Skills)

Slash commands live in `.claude/commands/`. Type `/command-name <args>` in
the Claude Code chat to invoke them. They make Claude act as a domain expert
**without any Python code running or API cost** — Claude reads the `.md` file
and responds in character.

This is different from calling the equivalent Python class directly.

### Available commands

| Command | File | What it does |
|---------|------|-------------|
| `/bridge-expert <claim>` | `.claude/commands/bridge-expert.md` | Validates a statistical bridge claim as a domain expert |

### `/bridge-expert` — when to use which interface

| Situation | Use this |
|-----------|----------|
| Quick sanity check during development ("does this number make sense?") | `/bridge-expert slam_rate=0.15 over 20 boards` |
| Supervisor asks a question mid-meeting | `/bridge-expert <paste their question>` |
| Automated validation after Stage 1 runs | `BridgeValidator().validate_profile_assignment(...)` in Python |
| Validate all 4 profiles at once programmatically | `BridgeValidator().validate_stage1_results(summary)` |

### `/bridge-expert` — example invocations

```
/bridge-expert slam_rate=0.101 over 216 declared boards — valid Slam Hunter?
/bridge-expert penalty_double_rate=0.40 over 15 boards — suspicious?
/bridge-expert Is nt_rate=0.385 with n=217 boards a strong NT Specialist?
/bridge-expert Player has n_declared=20 and slam_rate=0.20 — should we trust this?
```

The command knows NegoPlay baselines (slam ~5.5%, partscore ~57%, NT ~28.2%,
penalty double ~8.5%), Nezer's minimum (n≥50), and Duplicate Bridge rules.
Output follows the 4-part schema: Legality → Probability → Expert Analysis → Verdict.

### Adding new slash commands

Create `.claude/commands/<name>.md`. The file must end with the instructions
Claude should follow when the command is invoked. Use `$ARGUMENTS` as the
placeholder for user-provided text.

---

## 🛠️ Working with Antigravity

NegoPlay is built in Google Antigravity. When generating code, be aware:

### Antigravity-specific conventions
- **Artifacts** are auto-generated for each task (plans, diffs, walkthroughs).
  Suggest meaningful artifact titles.
- **Manager View** allows parallel agents. When suggesting work, indicate
  whether tasks should be sequential or parallelizable.
- **Browser agent** is available. Useful for: literature search, dataset
  exploration on Kaggle, checking API documentation live.
- **Trust policies** are set conservatively. Anna reviews artifacts before
  approving terminal commands.

### Workflow modes
- **Planning Mode** — for complex multi-file tasks (recommend this for
  Stages 1, 3, 4)
- **Fast Mode** — for quick iterations (prompt tweaks, doc updates)

### Agent suggestions
When proposing changes, include:
1. **Scope** — files affected
2. **Dependencies** — what must be done first
3. **Verification** — how to test it worked
4. **Cost estimate** — for any task involving Gemini API calls

---

## 📁 Code Organization

### Architectural Principle: **Single SDK Entry Point**

Per Dr. Segal's methodology framework (course requirement):

> *"No function lives only in a screen, script, or notebook. The SDK is the
> single contract. All GUI, CLI, and API calls go through it."*

Every reusable operation must live in `src/sdk.py` or `src/shared/`.

### Folder Structure
```
src/
├── sdk.py                    ← MAIN ENTRY POINT — public API
├── stage1_clustering/
│   ├── __init__.py
│   ├── features.py           ← Feature engineering
│   ├── clustering.py         ← K-Means, HDBSCAN
│   └── validation.py         ← Silhouette, p-values
├── stage2_skills/
│   ├── __init__.py
│   ├── chunker.py            ← Game chunking for LLM
│   ├── extractor.py          ← Gemini skill extraction
│   └── aggregator.py         ← Cross-chunk aggregation
├── stage3_agents/
│   ├── __init__.py
│   ├── base_agent.py         ← BaseAgent class
│   ├── bridge_agent.py       ← Bridge-playing agent
│   └── nego_agent.py         ← Negotiation agent
├── stage4_simulate/
│   ├── __init__.py
│   ├── bridge_game.py        ← Bridge auction simulation
│   ├── negotiation.py        ← Business scenario simulation
│   └── alignment.py          ← Cross-domain alignment analysis
├── shared/
│   ├── llm_client.py         ← Unified LLM wrapper (Gemini default, Claude/OpenAI optional)
│   ├── bridge_validator.py   ← Bridge Expert Validation Skill (statistical sanity checker)
│   ├── data_loader.py        ← CSV loading utilities
│   ├── prompts.py            ← Centralized prompt library
│   └── logger.py             ← Structured logging
└── report.py                 ← Final report generation
```

---

## 🎨 Code Style Rules

### Python style
- **Formatter:** `ruff format` (replaces `black`)
- **Linter:** `ruff check`
- **Type hints:** Required for all public functions
- **Docstrings:** Google style, required for all classes and public functions
- **Line length:** 100 characters max

### Naming conventions
```python
# Variables and functions: snake_case
player_features = compute_features(player_id)

# Classes: PascalCase
class BridgeAgent:
    pass

# Constants: UPPER_SNAKE_CASE
MIN_BOARDS_PER_PLAYER = 20
DEFAULT_K_CLUSTERS = 4
```

### Forbidden patterns
- ❌ No global mutable state
- ❌ No `from module import *`
- ❌ No bare `except:` (always specify exception)
- ❌ No `print()` for logging — use `logging` module
- ❌ No hardcoded API keys (use `.env`)
- ❌ No magic numbers — use named constants

---

## 🤖 Working with LLMs in This Project

### Model selection policy

Gemini is the default (cost). Claude / OpenAI are allowed for cross-model
validation or quality-critical tasks. All calls go through
`src/shared/llm_client.py` with a `provider=` argument.

| Task | Default model | Alt. provider OK? | Reason |
|------|---------------|-------------------|--------|
| Skill extraction | `gemini-2.0-flash-exp` | Yes (validation runs) | Cheap, large context |
| Bridge agents | `gemini-2.0-flash-exp` | Yes — useful to compare profile behavior across providers | Fast, consistent |
| Negotiation agents | `gemini-2.0-flash-exp` | Yes — same as above | Same as above |
| Final synthesis / writing | `gemini-2.5-pro` *or* `claude-opus-4-7` *or* `gpt-5` | Yes | Quality > cost |
| Cross-model robustness check | run on all 3 | Required | Strengthens publishability |

### Token economics rules

1. **Always chunk** when sending bridge games to LLM (20-30 games max per call)
2. **Cache prompts** — Gemini supports caching for repeated system prompts
3. **Structured output**: Use `response_mime_type="application/json"` with
   a schema. Never request free text.
4. **Log every call**: Token counts, latency, cost — saved to
   `results/llm_logs/`
5. **Budget alert**: If running cost > $20, stop and review (well below
   $50 hard cap)

### Standard Gemini call pattern

> ⚠️ **Deprecation warning (May 2026):** `google-generativeai` is deprecated.
> New code must use `google-genai` (`from google import genai`).
> Existing code in `bridge_validator.py` uses the old package — migrate when
> building `llm_client.py`.

```python
# NEW pattern (google-genai)
from google import genai
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=user_prompt,
    config={
        "system_instruction": PROFILE_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.3,   # bridge: 0.3, negotiation: 0.7
    },
)

result = json.loads(response.text)

# ALWAYS log the call
log_llm_call(
    model="gemini-2.0-flash",
    input_tokens=response.usage_metadata.prompt_token_count,
    output_tokens=response.usage_metadata.candidates_token_count,
    purpose="profile_extraction",
)
```

### Prompt principles

Every system prompt for an agent must include:
1. **Profile identity** (e.g., "You are a Slam Hunter")
2. **Core skills** (5-7 traits)
3. **Decision constraints** (e.g., bidding rules in bridge)
4. **Output format** (always structured JSON)
5. **Few-shot examples** (1-2 examples in the prompt)

---

## ✅ Testing Requirements

### Minimum coverage
- Stage 1 (clustering): 80%+ coverage
- Stages 2-4 (agents): 60%+ coverage (LLM calls hard to test)
- Shared utilities: 90%+ coverage

### Test naming
```python
def test_<unit>_<scenario>_<expected>():
    """E.g., test_clustering_with_k4_returns_4_groups"""
    pass
```

### Required test types
- Unit tests for all feature engineering
- Integration test for full Stage 1 pipeline
- Smoke test for each agent (one Gemini call each)
- E2E test that runs 1 bridge game (mocked Gemini)

### Mocking Gemini
```python
from unittest.mock import patch, MagicMock

@patch("google.generativeai.GenerativeModel")
def test_skill_extractor(mock_model):
    mock_response = MagicMock()
    mock_response.text = '{"skills": ["aggressive", "risk-taking"]}'
    mock_model.return_value.generate_content.return_value = mock_response
    # ... test logic
```

---

## 🐛 Common Issues & Debugging

### Issue: Gemini gives inconsistent responses
**Solution:** Set `temperature=0.3` for bridge bidding (consistency),
`temperature=0.7` for negotiation (variety). Use `response_mime_type="application/json"`.

### Issue: K-Means produces unstable clusters
**Solution:** Run with `random_state=42`, also try `n_init=20`.
Validate with HDBSCAN.

### Issue: Hebrew text in data causes encoding errors
**Solution:** Always read CSV with `encoding='utf-8-sig'`.

### Issue: Player names mismatched across tournaments
**Solution:** Use fuzzy matching from `src/shared/data_loader.py`.
This is a known issue.

### Issue: Gemini rate limit hit
**Solution:** Free tier = 15 req/min. Use `tenacity` with exponential
backoff. Consider upgrading to paid tier ($0 minimum, pay-as-you-go).

### Issue: Antigravity agent stops mid-task
**Solution:** Check the Artifacts panel — likely waiting for review.
Approve or modify in Manager View.

---

## 📋 Workflow Conventions

### Git commits
- **Format:** `<type>(<scope>): <description>`
- **Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- **Examples:**
  - `feat(stage1): add HDBSCAN clustering`
  - `fix(agents): correct slam hunter bidding logic`
  - `docs(readme): update architecture diagram`

### Branch naming
- `feature/<name>` — new features
- `fix/<name>` — bug fixes
- `experiment/<name>` — experimental work

### Solo project rules
This is a solo project, but follow these principles:
1. Don't push directly to `main` for major features
2. Keep commits small and focused (one logical change per commit)
3. Use meaningful commit messages (future Anna will thank present Anna)
4. Commit at least once per Antigravity session

---

## 🎯 What AI Assistants Should Do

### When Anna asks for code:
1. **Read PRD.md first** to understand requirements
2. **Read TASKS.md** to know current sprint
3. **Provide complete, copy-pasteable code blocks** (Anna's preference)
4. **Include type hints and docstrings**
5. **Explain decisions briefly** — Anna is technical, she wants the *why*
6. **Suggest tests** for new functions

### When Anna asks for explanations:
1. **Start with the intuition** (Anna's pedagogy is "why before how")
2. **Use simple examples** before diving into math
3. **Reference course materials** when relevant
4. **Respond in Hebrew if asked in Hebrew**, English otherwise

### When Anna seems stuck:
1. **Ask clarifying questions** instead of guessing
2. **Suggest the smallest next step** (avoid analysis paralysis)
3. **Reference TASKS.md** to refocus

### When generating prompts for the NegoPlay agents:
1. **Always include profile name** and 5-7 skills
2. **Always specify output format** (JSON with schema)
3. **Include 1-2 examples** in the prompt (few-shot)
4. **Test the prompt with 1 manual call** before scaling

---

## 🚨 Critical Don'ts

1. ⚠️ **Default to Gemini for cost.** Only reach for Claude or OpenAI when
   there's a concrete reason (cross-model validation, quality-critical
   synthesis, or a capability Gemini measurably lacks for the task at hand).
   Never hardcode a provider — always route through `llm_client.py`.

2. ❌ **Don't suggest changing the research direction.** Anna spent significant
   time aligning the project with her PhD. Stay within scope.

3. ❌ **Don't propose libraries outside the approved stack** without
   justification.

4. ❌ **Don't generate code that calls APIs without `.env` keys** — always
   use `os.getenv()` patterns.

5. ❌ **Don't write to `data/raw/`** — it's read-only. Always write to
   `data/processed/` or `results/`.

6. ❌ **Don't suggest fine-tuning LLMs.** It's explicitly out of scope.

7. ❌ **Don't generate fake/mock data without explicit permission.** The
   whole point is empirical work on real data.

8. ❌ **Don't bypass the SDK.** All public operations go through `src/sdk.py`.

---

## 📞 When Stuck — Escalation Path

1. Try one approach. If it fails, articulate the failure mode.
2. Suggest two alternatives with trade-offs.
3. Ask Anna to choose, or to provide more context.
4. Never assume — always ask when uncertain.

---

## 📝 Document Updates

This file should be updated when:
- New libraries are added to the stack
- New patterns emerge in the codebase
- Anna discovers a recurring issue worth documenting
- Major architectural decisions are made

**Last updated:** 2026-05-28 — Added slash commands section, bridge_validator.py,
Gemini SDK deprecation warning (google-generativeai → google-genai)
**Maintained by:** Anna Ben-Shushan + AI collaboration
