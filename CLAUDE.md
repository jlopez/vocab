# Project Guidelines

## Code Style
- Use typings throughout; code must pass mypy and ruff
- Use `list[]`, `dict[]` rather than `List[]`, `Dict[]`
- Avoid `any` unless absolutely necessary
- Include docstrings in every public-facing method (not required in tests)
- Use src-layout: code in `src/`, tests in `tests/`

## Test Coverage
- Aim for 100% coverage when practical; minimum threshold is 90%
- Don't do excessive work to cover edge cases that provide little value

## Coverage Badge
When coverage changes, update the badge in README.md:
- 90%+ → `brightgreen`, 80-89% → `green`, 70-79% → `yellowgreen`
- 60-69% → `yellow`, 50-59% → `orange`, <50% → `red`

Format: `![Coverage](https://img.shields.io/badge/coverage-XX%25-COLOR)`

## Running Commands
- Always use `uv run` to ensure the virtualenv is used
- Examples: `uv run pytest`, `uv run mypy .`, `uv run python script.py`

## Dependencies
- Add dependencies using `uv add <package>` (use `--dev` for dev dependencies)

## Git Commits
- Use conventional commits: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

## Code Reviews
Conduct **adversarial code reviews** unless otherwise specified:
- Assume the code under review is in the last git commit
- Focus on bugs, edge cases, security issues, and design flaws
- Organize findings by severity (Critical, Moderate, Minor) w/unique numbering
- Include file paths and line numbers for each issue

## Implementation Planning

Use `docs/plan.md` for multi-phase implementation work.

### When Asked to Plan
1. Discuss the approach with the user to reach shared understanding
2. Write the plan to `docs/plan.md` with:
   - Overview and architecture
   - Phases broken into single-session chunks
   - Clear deliverables and acceptance criteria per phase
   - Files to create, modify, and delete
3. Update README.md if public API or usage changes

### When Implementing a Phase
1. Read `docs/plan.md` to understand the current phase
2. Implement according to the deliverables and acceptance criteria
3. Run all checks (ruff, mypy, pytest with coverage)
4. Once passing: mark phase complete in plan.md, update README.md, commit

### Code Review During Planning
1. Compare implementation against plan's acceptance criteria
2. Conduct adversarial review (match plan? criteria met? bugs?)
3. Report gaps between plan and implementation

### Committing Code Review Fixes
- Amend the previous phase commit (`git commit --amend --no-edit`)
- Exception: if fixes substantially change scope, discuss with user first

### Continuing Work
Read `docs/plan.md`, check git status/log, continue from where previous session left off.

### Completing a Plan
When fully implemented and code reviewed:
1. Run final quality checks (ruff, mypy, pytest)
2. Verify README.md accuracy (features, examples, coverage badge, architecture)
3. Check plan.md for info to preserve (decisions → docstrings, future work → Issues)
4. Squash all phase commits into a single commit:
   - Use `git reset --soft <commit-before-first-phase>`
   - Write a commit message describing the feature as a whole (not individual phases)
   - The message should explain what the feature does and key implementation details
