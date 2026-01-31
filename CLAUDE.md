# Project Guidelines

## Code Style
- Use typings throughout
- Code should be free of any typing errors (`mypy src`) and free of any ruff errors
- Do not use `any` unless absolutely necessary
- Use `list[]`, `dict[]` rather than `List[]`, `Dict[]`
- Include docstrings in every public-facing method (not required in tests)

## Project Structure
- Use src-layout: code in `src/`
- Write tests in `tests/`

## Test Coverage
- Aim to maintain 100% test coverage when practical
- Don't do excessive work to cover edge cases that provide little value
- Minimum threshold is 90% (enforced in pyproject.toml)

## Coverage Badge
When test coverage changes, update the badge in README.md. Use this color scale:
- 90%+ → `brightgreen`
- 80-89% → `green`
- 70-79% → `yellowgreen`
- 60-69% → `yellow`
- 50-59% → `orange`
- <50% → `red`

Format: `![Coverage](https://img.shields.io/badge/coverage-XX%25-COLOR)`

## Running Commands
- Always use `uv run` to run Python, pytest, mypy, etc. to ensure the virtualenv is used
- Examples: `uv run pytest`, `uv run mypy src`, `uv run python script.py`

## Dependencies
- Add dependencies using `uv add <package>`, not by editing pyproject.toml directly
- Add dev dependencies using `uv add --dev <package>`

## Git Commits
- Use conventional commits format: `type(scope): description`
- Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

## Warnings and Logging
- Use the `warnings` module (not `logging` or `print`) for non-fatal issues that callers may want to handle
- This allows callers to filter/suppress warnings with `warnings.filterwarnings()`

## Code Reviews
When asked for a code review, conduct an **adversarial code review** unless otherwise specified:
- Assume the code under review is in the last git commit
- Focus on finding bugs, edge cases, security issues, and design flaws
- Be critical and thorough - the goal is to find problems, not validate the code
- Organize findings by severity (Critical, Moderate, Minor)
- Include file paths and line numbers for each issue
- Note design observations that aren't bugs but may warrant discussion

## Implementation Planning

Use `docs/plan.md` for multi-phase implementation work:

### When Asked to Plan
1. Discuss the approach with the user to reach shared understanding
2. Write the plan to `docs/plan.md` with:
   - Overview and architecture
   - Phases broken into single-session chunks
   - Clear deliverables and acceptance criteria per phase
   - Files to create, modify, and delete
   - Test requirements
3. Update `README.md` if CLI or usage changes

### When Implementing a Phase
1. Read `docs/plan.md` to understand the current phase
2. Implement according to the deliverables and acceptance criteria
3. Run all acceptance criteria checks (ruff, mypy, pytest, coverage)
4. Once all checks pass:
   - Update `docs/plan.md` to mark the phase as complete (add checkmarks to acceptance criteria)
   - Commit all changed files with a conventional commit message

### When Asked for Code Review (During Planning)
1. Read `docs/plan.md` to understand what was supposed to be implemented
2. Compare the implementation against the acceptance criteria
3. Conduct adversarial review focusing on:
   - Does the implementation match the plan?
   - Are acceptance criteria actually met?
   - Bugs, edge cases, security issues
4. Report any gaps between plan and implementation

### Committing Code Review Fixes
When the user is satisfied with the code review fixes:
1. Amend the previous phase commit (`git commit --amend --no-edit`)
2. Do not change the commit message - fixes are refinements, not new scope
3. Exception: if the fixes substantially change what was implemented, discuss with the user before updating the commit message

### Continuing Work in New Sessions
When resuming work on a plan:
1. Read `docs/plan.md` to understand overall context and current phase
2. Check git status/log to see what's already been done
3. Continue from where the previous session left off
