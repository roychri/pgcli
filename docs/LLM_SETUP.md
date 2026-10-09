# LLM Provider Setup Guide for pgcli

This guide explains how to set up and configure different LLM providers for pgcli's natural language SQL generation feature.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Anthropic (Claude) Setup](#anthropic-claude-setup)
3. [OpenAI (GPT) Setup](#openai-gpt-setup)
4. [Ollama (Local) Setup](#ollama-local-setup)
5. [Provider Comparison](#provider-comparison)
6. [Environment Variables Reference](#environment-variables-reference)
7. [Advanced Configuration](#advanced-configuration)

---

## Quick Start

**Install pgcli with LLM support:**

```bash
pip install -U "pgcli[llm]"
```

This installs:
- `pydantic-ai-slim` with the Anthropic and OpenAI clients - Multi-provider LLM framework
  (Ollama uses the OpenAI-compatible client)
- `tiktoken` - Token counting for OpenAI models

**Choose your provider:**

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-..."
pgcli postgresql://localhost/mydb

# OpenAI
export PGCLI_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
pgcli postgresql://localhost/mydb

# Ollama (local)
export PGCLI_LLM_PROVIDER=ollama
export PGCLI_LLM_MODEL=llama3.2
pgcli postgresql://localhost/mydb
```

Press **F6** to toggle LLM mode, then type your query in natural language!

---

## Anthropic (Claude) Setup

**Default provider** - Claude models are optimized for SQL generation and reasoning.

### 1. Get an API Key

Visit: https://console.anthropic.com/

1. Sign up or log in
2. Go to "API Keys"
3. Create a new key
4. Copy the key (starts with `sk-ant-...`)

### 2. Set Environment Variable

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

**Make it permanent:**

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.bashrc
source ~/.bashrc
```

### 3. Choose a Model (Optional)

**Default:** `claude-sonnet-5-5` (Claude Sonnet 5.5)

**Current models:**
- `claude-opus-5-5` - Most capable, for long, complex work ($4/$20 per million tokens)
- `claude-sonnet-5-5` - Best balance of speed and intelligence ($2/$10)
- `claude-haiku-5-5` - Fastest and cheapest ($0.10/$0.50)

Any model name pydantic-ai supports works, including older ones such as `claude-sonnet-4-5`.

**To change model:**

```bash
export PGCLI_LLM_MODEL="claude-opus-5-5"
```

### 4. Verify Setup

```bash
pgcli postgresql://localhost/mydb

# Press F6 to enable LLM mode
# Status bar should show: [F6] LLM Mode: ON (0 tokens)
```

### Features

- ✅ **Context Window:** 1M tokens (pgcli sends at most 180K of history by default)
- ✅ **Function Calling:** Excellent (optimized for tools)
- ✅ **Thinking:** Enabled at `low` effort by default (see `PGCLI_LLM_THINKING`)
- ✅ **Cost:** $2/$10 per million tokens (input/output) for Sonnet 5.5

---

## OpenAI (GPT) Setup

OpenAI models (GPT-4o, GPT-4o-mini) are fast and reliable for SQL generation.

### 1. Get an API Key

Visit: https://platform.openai.com/

1. Sign up or log in
2. Go to "API keys"
3. Create a new secret key
4. Copy the key (starts with `sk-proj-` or `sk-...`)

### 2. Set Environment Variables

```bash
export PGCLI_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-proj-..."
```

**Make it permanent:**

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PGCLI_LLM_PROVIDER=openai' >> ~/.bashrc
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
source ~/.bashrc
```

### 3. Choose a Model (Optional)

**Default:** `gpt-6-luna` (efficient, low cost)

**Current models:**
- `gpt-6-luna` - Efficient model for focused, high-volume tasks ($0.10/$0.50 per million tokens)
- `gpt-6-sol` - Complex coding and agentic workflows ($2/$10)
- `gpt-6.1-sol` - Newer Sol, lower cost than Astra for professional work ($2/$10)
- `gpt-6-astra` - Most capable, for the hardest end-to-end work ($10/$50)

Any model name pydantic-ai supports works, including older ones such as `gpt-4o-mini`.
OpenAI models are called through the Responses API.

**To change model:**

```bash
export PGCLI_LLM_MODEL="gpt-6-sol"
```

### 4. Verify Setup

```bash
pgcli postgresql://localhost/mydb

# Press F6 to enable LLM mode
# Type a query in natural language
```

### Features

- ✅ **Context Window:** 1.05M tokens (pgcli sends at most 180K of history by default)
- ✅ **Function Calling:** Excellent
- ✅ **Reasoning:** Enabled at `low` effort by default (see `PGCLI_LLM_THINKING`)
- ✅ **Cost:** $0.10/$0.50 per million tokens (gpt-6-luna)

---

## Ollama (Local) Setup

Run LLMs locally on your machine (no API key needed, no internet required).

### 1. Install Ollama

Visit: https://ollama.ai/

```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Download a Model

```bash
# Recommended for SQL (3B parameters, fast)
ollama pull llama3.2

# Alternatives
ollama pull mistral      # 7B, very good
ollama pull codellama    # 7B, code-focused
ollama pull qwen2.5      # 7B, good reasoning
```

### 3. Start Ollama Server

```bash
ollama serve
```

Leave this running in a terminal.

### 4. Configure pgcli

```bash
export PGCLI_LLM_PROVIDER=ollama
export PGCLI_LLM_MODEL=llama3.2  # or your chosen model
```

### 5. Verify Setup

```bash
pgcli postgresql://localhost/mydb

# Press F6 to enable LLM mode
```

### Features

- ✅ **Privacy:** Everything runs locally
- ✅ **Cost:** Free (no API charges)
- ✅ **Internet:** Not required after download
- ⚠️ **Context Window:** 2,000-8,000 tokens (model dependent, small!)
- ⚠️ **Quality:** Good but not as good as Claude/GPT
- ⚠️ **Speed:** Depends on your hardware

### Recommendations

**Best local models for SQL:**
1. `llama3.2` - Fast, decent quality
2. `mistral` - Good balance
3. `codellama` - Better for complex SQL

**Hardware requirements:**
- Minimum: 8GB RAM, 4GB VRAM
- Recommended: 16GB RAM, 8GB VRAM

---

## Provider Comparison

| Feature | Anthropic (Claude) | OpenAI (GPT) | Ollama (Local) |
|---------|-------------------|--------------|----------------|
| **Context Window** | 1M tokens | 1.05M tokens | 2K-8K tokens |
| **History Sent** | up to 180K | up to 180K | 1.8K-7.2K (90%) |
| **SQL Result Limit** | 18K tokens | 18K tokens | 180-720 tokens |
| **Function Calling** | Excellent | Excellent | Good |
| **SQL Quality** | Excellent | Very Good | Good |
| **Speed** | Fast | Very Fast | Variable |
| **Cost (per M tokens)** | $2/$10 (Sonnet) | $0.10/$0.50 (Luna) | Free |
| **Privacy** | Cloud | Cloud | Local |
| **Setup Difficulty** | Easy | Easy | Medium |
| **Internet Required** | Yes | Yes | No (after setup) |
| **Default Model** | claude-sonnet-5-5 | gpt-6-luna | llama3.2 |

### Which to Choose?

**Use Anthropic (Claude) if:**
- You want the best SQL generation quality
- You need long conversation history (large context)
- You're okay with cloud-based processing
- **Recommended for production use**

**Use OpenAI (GPT) if:**
- You want fast responses
- You prefer lower costs (gpt-6-luna)
- You're already using OpenAI services
- **Good for high-volume usage**

**Use Ollama if:**
- Privacy is critical (healthcare, finance)
- You have no internet or want offline capability
- You want zero ongoing costs
- You have good hardware
- **Best for development/testing**

---

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Claude) | `sk-ant-api03-...` |
| `OPENAI_API_KEY` | OpenAI API key (if using GPT) | `sk-proj-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PGCLI_LLM_PROVIDER` | LLM provider to use | `anthropic` | `openai`, `ollama` |
| `PGCLI_LLM_MODEL` | Model name (provider-specific) | See defaults | `claude-opus-5-5`, `gpt-6-sol`, `llama3.2` |
| `PGCLI_LLM_THINKING` | Thinking/reasoning effort | `low` | `off`, `on`, `default`, `minimal`, `medium`, `high`, `xhigh` |
| `PGCLI_LLM_MAX_CONTEXT_TOKENS` | Max tokens of history sent per request | 90% of the model's window, capped at 180K | `50000`, `500000` |
| `PGCLI_LLM_LOG_DIR` | Directory for LLM interaction logs | None | `/tmp/pgcli-llm` |
| `PGCLI_LLM_API_KEY` | Generic API key (fallback) | None | `your-key` |

### Provider-Specific Defaults

**Anthropic:**
- Model: `claude-sonnet-5-5`
- API Key from: `ANTHROPIC_API_KEY`

**OpenAI:**
- Model: `gpt-6-luna`
- API Key from: `OPENAI_API_KEY`

**Ollama:**
- Model: `llama3.2`
- API Key: Not required
- URL: `http://localhost:11434`

---

## Advanced Configuration

### Custom Context Limits

By default pgcli sends at most 90% of the model's context window, capped at 180K
tokens, since resending a million tokens of history with every question is slow and
costly. Context windows come from pydantic-ai's model data (`genai-prices`).
Override the limit:

```bash
export PGCLI_LLM_MAX_CONTEXT_TOKENS=50000   # smaller and cheaper
export PGCLI_LLM_MAX_CONTEXT_TOKENS=500000  # use more of a 1M-token window
```

**Use cases:**
- Reserve more tokens for LLM response
- Limit costs by reducing context
- Test trimming behavior

### Multiple API Keys

Use different keys for different databases:

```bash
# Production database with Claude
ANTHROPIC_API_KEY="sk-ant-prod..." pgcli prod_db

# Dev database with local Ollama
PGCLI_LLM_PROVIDER=ollama pgcli dev_db
```

### Configuration File

Create `~/.config/pgcli/llm_config.env`:

```bash
# LLM Configuration
PGCLI_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...
PGCLI_LLM_MODEL=claude-sonnet-5-5
```

Then source it:

```bash
source ~/.config/pgcli/llm_config.env
pgcli
```

### Logging for Debugging

Enable debug logging to see token counts and trimming:

```bash
# Add to ~/.config/pgcli/config
[main]
log_file = /tmp/pgcli.log

# Then check logs
tail -f /tmp/pgcli.log
```

You'll see messages like:
```
DEBUG:pgcli.llm.agent:SQL result added to history: 14392 tokens (not trimmed)
DEBUG:pgcli.llm.agent:History trimmed: 30 msgs (185000 tokens) → 22 msgs (160000 tokens)
```

---

## Next Steps

- Read [LLM_USAGE.md](LLM_USAGE.md) - How to use LLM mode effectively
- Check [Troubleshooting](#troubleshooting) if you encounter issues
- See [PHASE3_FINDINGS.md](../PHASE3_FINDINGS.md) - Technical implementation details

## Support

If you encounter issues:

1. Check the [Troubleshooting Guide](LLM_USAGE.md#troubleshooting)
2. Enable debug logging (see above)
3. Report issues: https://github.com/dbcli/pgcli/issues
4. Include: Provider, model, error message, and relevant logs
