# InvenTree Telegram Bot

A Telegram chatbot that manages a personal home [InvenTree](https://inventree.org/) inventory system using Google Gemini AI as the backend.

Send text queries or photos to the bot and it will search your inventory, identify items, suggest where to put things, and create/update entries — all through natural conversation.

## Features

- **Natural language queries** — "Where are my AA batteries?", "How many USB cables do I have?"
- **Photo recognition** — Send a photo and the bot identifies the item, searches for matches, and suggests how to catalogue it
- **Smart placement** — "Where should I put this?" suggests locations based on existing inventory structure
- **Full CRUD** — Create parts, stock items, categories, and locations through conversation
- **Conversation memory** — Maintains per-user chat history with automatic compaction when the context budget is reached
- **Persistent sessions** — Conversation summaries survive container restarts
- **Dual compaction strategy** — Hot compaction during active chat (keeps recent messages raw), idle compaction after inactivity (summarises everything)
- **Model fallback chains** — Automatically falls back to the next model on rate limits (429)
- **Free tier friendly** — Designed for Gemini's free quota constraints (500 RPD lite, 20 RPD flash)

## Architecture

```
Telegram <-> Bot (python-telegram-bot)
                |
                +-> Gemini AI (function calling)
                |       |
                |       +-> InvenTree REST API (httpx)
                |
                +-> Per-user session management
                +-> Inventory context snapshots
```

**Key design trade-off**: Tokens are plentiful (250K TPM), requests are scarce (5-20 RPD for flash models). The architecture spends tokens generously (rich system prompts, conversation history, inventory snapshots) to minimise the number of API rounds needed.

## Project Structure

```
.
├── sample-data/
│   ├── prompt.txt              # Base system prompt (seeded to data/ on first run)
│   └── prompt-site-example.txt # Example site-specific prompt extension
├── data/                       # Runtime data (gitignored, volume-mounted)
│   ├── prompt.txt              # Active base prompt (editable without rebuild)
│   ├── prompt-*.txt            # Site-specific prompt extensions (see below)
│   ├── context.txt             # Inventory snapshot (auto-refreshed)
│   └── sessions/               # Per-user conversation state
│       └── {user_id}.json
├── src/
│   ├── bot.py              # Telegram handlers and command registration
│   ├── agent.py            # Gemini agent with function calling and fallback
│   ├── session.py          # Per-user conversation history and compaction
│   ├── compaction.py       # Inventory context snapshots and data dir init
│   ├── config.py           # Pydantic settings (env vars)
│   └── inventree_client.py # InvenTree REST API client
├── docker-compose.yml      # Development compose (builds from source)
├── Dockerfile
├── requirements.txt
└── .env.example
```

### System Prompt

The system prompt is assembled from multiple files in `data/`:

- **`prompt.txt`** — Base instructions (seeded from `sample-data/prompt.txt` on first run, updated with new code releases)
- **`prompt-*.txt`** — Site-specific extensions, sorted by name and appended to the base prompt. These are gitignored and specific to each deployment.

For example, create `data/prompt-hochzoll-126.txt` with your apartment layout, room naming conventions, and furniture prefixes. See `sample-data/prompt-site-example.txt` for a template.

This separation means `prompt.txt` evolves with code (new capabilities, decision logic) while site config stays untouched across updates.

## Deployment

This bot is designed to run alongside InvenTree on the same Docker network.

### Prerequisites

- An InvenTree instance (with an API token)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- A Google Gemini API key (free tier from [AI Studio](https://aistudio.google.com/apikey))

### Setup

1. **Clone and configure**

   ```bash
   git clone git@github.com:guyiyu/inventree-telegram-bot.git
   cd inventree-telegram-bot
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build and run (development)**

   ```bash
   docker compose up -d
   ```

   The `docker-compose.yml` in the repo builds from source and is suitable for development.

3. **Production deployment (separate build and run)**

   Build the image from the project directory:

   ```bash
   cd ~/Projects/inventree-telegram-bot
   git pull
   docker build -t inventree-telegram-bot .
   ```

   Run from a separate services directory with just `.env`, `data/`, and a compose file:

   ```yaml
   # ~/Services/inventree-telegram-bot/docker-compose.yml
   services:
     bot:
       image: inventree-telegram-bot
       container_name: inventree-telegram-bot
       restart: unless-stopped
       env_file:
         - .env
       volumes:
         - ./data:/app/data
       networks:
         - inventree

   networks:
     inventree:
       name: ${INVENTREE_NETWORK:-inventree_default}
       external: true
   ```

   ```bash
   cd ~/Services/inventree-telegram-bot
   docker compose up -d
   ```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token |
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `INVENTREE_URL` | No | `http://inventree-server:8000` | InvenTree API URL (internal Docker) |
| `INVENTREE_SITE_URL` | No | `https://inventree.example.com` | InvenTree web UI URL (for links) |
| `INVENTREE_API_TOKEN` | Yes | — | InvenTree API token |
| `ALLOWED_USER_IDS` | No | (empty = allow all) | Comma-separated Telegram user IDs |
| `GEMINI_MODELS_TEXT` | No | `gemini-flash-lite-latest` | Text model fallback chain |
| `GEMINI_MODELS_VISION` | No | `gemini-flash-latest,gemini-flash-lite-latest` | Vision model fallback chain |
| `CONTEXT_BUDGET` | No | `32000` | Max tokens for conversation context |
| `CONTEXT_REFRESH_INTERVAL` | No | `300` | Inventory snapshot refresh interval (seconds) |

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message |
| `/status` | Show API usage, model info, and session stats |
| `/clear` | Clear conversation history and summary |
| `/user_id` | Show your Telegram user ID (works for unauthenticated users) |

## License

MIT
