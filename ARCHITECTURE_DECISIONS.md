# Architecture Decisions - Phase 0
## pgcli Pydantic Migration

**Date:** 2025-09-30
**Status:** In Progress

---

## Decision Log

### ✅ DECIDED: Default LLM Provider
**Decision:** Anthropic (Claude)
**Rationale:**
- Best reasoning capabilities among available providers
- Excellent long-context handling (200K tokens)
- Strong performance on code and technical tasks
- Reliable function calling support

**Decided by:** Christian Roy
**Date:** 2025-09-30

---

### ✅ DECIDED: Default Model
**Decision:** Claude Sonnet 4.5 (claude-sonnet-4-20250514)
**Rationale:**
- Latest Sonnet model with cutting-edge performance
- Superior reasoning for SQL query generation
- Excellent balance of speed, quality, and cost
- Advanced function calling capabilities

**Decided by:** Christian Roy
**Date:** 2025-09-30

---

### ✅ DECIDED: API Key Storage
**Decision:** .env file (with fallback to environment variables)
**Rationale:**
- Standard Python practice (python-dotenv)
- Easy for development and deployment
- Can be gitignored for security
- Familiar to developers

**Implementation:**
- Primary: `.env` file in project root
- Fallback: Environment variable `ANTHROPIC_API_KEY`
- Retrieval command: `skate get anthropic_api_key`

**Decided by:** Christian Roy
**Date:** 2025-09-30

---

### ✅ DECIDED: Confirmation Strategy for Destructive Queries

**Decision:** Destructive queries FORBIDDEN by default, opt-in to enable

**Behavior:**
- Default: Block all destructive queries (DROP, DELETE, TRUNCATE, UPDATE without WHERE)
- Error message: "Destructive queries are disabled by default for safety"
- Enable via:
  - Environment variable: `PGCLI_ALLOW_DESTRUCTIVE=true`
  - CLI parameter: `--allow-destructive`
  - Config file: `[llm] allow_destructive = true`

**Rationale:**
- Maximum safety - prevents accidental data loss
- Explicit opt-in makes users aware of risk
- Can be enabled per-session or permanently
- Clear error message educates users

**Decided by:** Christian Roy
**Date:** 2025-09-30

---

### ⚠︎  PENDING: Configuration Format

**Options:**

1. **Extend pgclirc (Recommended)**
   - Add `[llm]` section to existing config
   - Familiar to pgcli users
   - Pros: One config file, consistent UX
   - Cons: Need to extend config parser

2. **Separate llm.conf**
   - New file just for LLM settings
   - Pros: Separation of concerns
   - Cons: Another config file to manage

**Recommendation:** Option 1 (Extend pgclirc)

**Status:** Assumed decided per migration plan, confirm if needed

---

### ⚠︎  PENDING: Offline Support Strategy

**Options:**

1. **Require Ollama for Offline**
   - Users install Ollama locally
   - Configure provider=ollama in pgclirc
   - Pros: True offline capability
   - Cons: Additional setup burden

2. **Graceful Degradation**
   - LLM features disabled without API access
   - Show friendly message
   - Pros: Simple, no surprises
   - Cons: Feature unavailable offline

3. **Hybrid (Recommended)**
   - Detect Ollama installation automatically
   - Offer to use Ollama if API key missing
   - Fall back to disable if neither available
   - Pros: Best UX, automatic optimization
   - Cons: More complex logic

**Recommendation:** Option 3 (Hybrid)
- Phase 1: Implement Option 2 (graceful degradation)
- Phase 2-3: Add Option 3 (Ollama detection and fallback)

**Decision needed from:** Christian Roy (can decide later in Phase 2)

---

### ⚠︎  PENDING: Education Features Scope

**Options:**

1. **Full Education Mode**
   - explain_concept tool
   - Learning journeys
   - SQL quizzes
   - Skill level tracking

2. **Basic Explanations Only**
   - Just explain queries on request
   - No journeys or quizzes

3. **Phase-based Approach (Recommended)**
   - Phase 0-2: Basic explanations
   - Phase 3: Full education features

**Recommendation:** Option 3 (Phase-based)

**Status:** Can decide during Phase 3 planning

---

### ⚠︎  PENDING: Multi-Database Support

**Question:** Should we plan for MySQL/SQLite support eventually?

**Options:**

1. **PostgreSQL Only (Recommended for Phase 0-3)**
   - Focus on excellence for PostgreSQL
   - Simplifies implementation
   - Matches pgcli's purpose

2. **Database-Agnostic Architecture**
   - Design tools to work with any SQL database
   - More complex initial implementation
   - Enables future expansion

**Recommendation:** Option 1 for now, Option 2 architecture in mind
- Implement for PostgreSQL only
- Design tools with database abstraction where easy
- Revisit in post-launch roadmap

**Status:** Not urgent, can decide in Phase 2-3

---

## What We Need to Proceed with Phase 0

### ✅ Confirmed Available
- [x] Anthropic API key (via `skate get anthropic_api_key`)
- [x] Development environment (WSL2, Python, git)
- [x] pgcli codebase (already on add-llm-support branch)

### ✅ ANSWERED: Questions for Christian

1. **Confirmation Strategy:** ✅ DECIDED - Destructive queries forbidden by default, opt-in to enable

2. **Test Database:** ✅ DECIDED - Use Docker to start PostgreSQL on non-standard port

3. **Other Providers:** ✅ DECIDED - Focus only on Anthropic for Phase 0 prototype

---

## Next Steps (After Task 1)

Once architecture decisions are documented:
- Task 2: ✅ Already decided (Anthropic)
- Task 3: Decide confirmation strategy (can be quick decision or prototype-first)
- Task 4: ✅ Already decided (.env file)
- Task 5: Create pydantic-migration branch
- Task 6-7: Install dependencies
- Task 8-13: Build prototype

---

## Notes

- Using Claude Sonnet 4.5 as default (latest Sonnet model - claude-sonnet-4-20250514)
- .env file pattern follows Python best practices
- Destructive queries forbidden by default - explicit opt-in required
- Docker PostgreSQL on non-standard port for testing
- Anthropic-only for Phase 0 prototype
- Phased approach allows iteration based on feedback
