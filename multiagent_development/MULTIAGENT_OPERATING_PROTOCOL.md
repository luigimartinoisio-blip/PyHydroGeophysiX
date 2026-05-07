# Multi-Agent Operating Protocol

## Purpose

This document defines the **operating rules and principles** for all AI agents working on PyHydroGeophysiX_Carl. These rules ensure coordinated, safe, and scientifically rigorous development.

## Core Operating Principles

### Principle 1: Read Before Acting
**Rule**: Always understand the current state before proposing changes.

**Implementation**:
- Read relevant source files completely
- Review existing tests
- Check documentation
- Understand data contracts
- Verify assumptions

**Rationale**: Prevents breaking existing functionality and ensures context-aware decisions.

### Principle 2: Propose Before Modifying
**Rule**: Never modify code directly without a formal change request.

**Implementation**:
- Create change request in `CHANGE_REQUESTS.md`
- Describe rationale and impact
- Identify affected components
- Propose testing strategy
- Wait for approval (human or orchestrator)

**Rationale**: Maintains control and traceability of all modifications.

### Principle 3: Test Before Integrating
**Rule**: Every code change must have associated tests.

**Implementation**:
- Write unit tests for new functions
- Write integration tests for new workflows
- Update existing tests if behavior changes
- Ensure all tests pass before marking task complete
- Document test coverage

**Rationale**: Ensures reliability and prevents regressions.

### Principle 4: Document Before Closing
**Rule**: Every feature must be documented before task completion.

**Implementation**:
- Update API documentation
- Update user guides if needed
- Update developer guides if architecture changes
- Add docstrings to new functions
- Update CHANGELOG

**Rationale**: Maintains project usability and knowledge transfer.

### Principle 5: Human Review for Critical Decisions
**Rule**: Scientific and architectural decisions require human approval.

**Implementation**:
- Add item to `HUMAN_REVIEW_QUEUE.md`
- Provide context and options
- Wait for human decision
- Document decision in `DECISIONS_LOG.md`
- Proceed with approved approach

**Rationale**: Preserves scientific integrity and expert judgment.

## Agent Coordination Rules

### Rule 1: Single Responsibility
Each agent has a defined scope. Do not perform tasks outside your domain.

**Example**:
- ✅ Hydrogeology Agent modifies Modflow integration
- ❌ Hydrogeology Agent modifies ERT forward modeling

**Action**: If task requires multiple domains, coordinate through Project Orchestrator.

### Rule 2: Task Assignment
Tasks must be explicitly assigned to an agent before work begins.

**Process**:
1. Project Orchestrator assigns task
2. Agent acknowledges assignment
3. Agent updates task status to "in_progress"
4. Agent completes work
5. Agent updates task status to "completed"
6. Agent notifies Orchestrator

### Rule 3: Artifact Production
Every task must produce a tangible artifact.

**Artifacts include**:
- Code changes (with tests)
- Documentation updates
- Analysis reports
- Design documents
- Test results
- Review comments

### Rule 4: Status Reporting
Agents must update status regularly.

**Required updates**:
- Task start: Update `TASK_BOARD.md`
- Blockers encountered: Update `TASK_BOARD.md` and notify Orchestrator
- Task completion: Update `TASK_BOARD.md` and relevant reports
- Daily: Update `reports/active_tasks_report.md`

### Rule 5: Dependency Management
Agents must declare dependencies before starting work.

**Process**:
1. Identify prerequisite tasks
2. Verify prerequisites are complete
3. If blocked, update status and wait
4. Do not proceed with incomplete dependencies

## Code Modification Rules

### Rule 1: No Direct Modifications to Tested Workflows
**Protected areas**:
- Existing `src/` modules with passing tests
- Data pipeline components
- Tested notebook workflows
- Configuration files (`pyproject.toml`, `pytest.ini`)

**Required process**:
1. Create change request
2. Get approval from relevant guardian agent
3. Implement in branch or isolated module
4. Test thoroughly
5. Get human review if scientifically significant
6. Integrate with full test suite passing

### Rule 2: Preserve Backward Compatibility
Breaking changes require explicit justification and migration path.

**Process**:
1. Document breaking change in change request
2. Provide migration guide
3. Update all affected code
4. Update all affected tests
5. Update all affected documentation
6. Get human approval

### Rule 3: Follow Existing Patterns
New code should match existing style and architecture.

**Guidelines**:
- Use existing naming conventions
- Follow existing module structure
- Match existing error handling patterns
- Use existing logging patterns
- Follow existing documentation style

### Rule 4: Incremental Changes
Prefer small, focused changes over large rewrites.

**Best practices**:
- One logical change per commit
- One feature per pull request
- Refactor separately from feature addition
- Test each increment independently

## Scientific Modification Rules

### Rule 1: Explicit Assumptions
All scientific assumptions must be documented.

**Process**:
1. Document assumption in `SCIENTIFIC_ASSUMPTIONS.md`
2. Mark as "verified", "assumed", or "to_verify"
3. Cite references if available
4. Update when verified or refuted

### Rule 2: Physical Constraints
Physical bounds must be enforced.

**Examples**:
- Porosity: 0 < φ < 1
- Saturation: 0 ≤ S ≤ 1
- Resistivity: ρ > 0
- Hydraulic conductivity: K > 0

**Implementation**:
- Add bounds checks in code
- Add tripwire alerts for violations
- Document bounds in parameter definitions

### Rule 3: Baseline Inversion Isolation
ERT baseline inversion must NOT enter optimization loop.

**Enforcement**:
- Baseline inversion used only for visualization
- Optimization compares apparent resistivities (data domain)
- No direct use of inverted resistivity in objective function
- Document this separation clearly

### Rule 4: Data-Domain Optimization
Optimization must operate in data domain, not model domain.

**Implementation**:
- Objective function: misfit between observed and simulated apparent resistivities
- Forward modeling: hydro state → true resistivity → apparent resistivity
- No direct comparison with inverted resistivity
- Document data flow clearly

### Rule 5: Tripwire Requirements
Safety checks are mandatory for optimization.

**Required tripwires**:
- Physical bounds violations
- Gradient stall detection
- Misfit stagnation detection
- Parameter identifiability issues
- Systematic spatial residuals
- Jacobian conditioning problems

**Action**: Tripwire activation triggers human review.

## Data Pipeline Rules

### Rule 1: Data Pipeline Guardian Approval
All data pipeline changes require Data Pipeline Guardian review.

**Scope**:
- Data ingestion
- Data connectors
- Temporal alignment
- Quality control
- Data transformations

### Rule 2: Preserve Existing Tests
Data pipeline changes must not break existing tests.

**Process**:
1. Run existing tests before changes
2. Implement changes
3. Run existing tests after changes
4. Fix any failures
5. Add new tests for new functionality

### Rule 3: Data Contract Stability
Changes to data contracts require formal approval.

**Process**:
1. Document proposed contract change
2. Identify all affected components
3. Propose migration strategy
4. Get Scientific Architect approval
5. Get human approval
6. Implement with full test coverage

## Quality Gate Rules

### Rule 1: Mandatory Quality Gates
All changes must pass defined quality gates before integration.

**Gates** (see `QUALITY_GATES.md`):
- G01: Change request exists
- G02: Tests exist and pass
- G03: Scientific rationale documented
- G04: Data pipeline tests preserved
- G05: Baseline inversion isolated
- G06: Optimization in data domain
- G07: Tripwires implemented
- G08: Dashboard is advisory only
- G09: Documentation updated
- G10: Human review completed (if required)

### Rule 2: No Gate Skipping
Quality gates cannot be bypassed without explicit human approval.

**Exception process**:
1. Document reason for exception
2. Assess risk
3. Get human approval
4. Document in `DECISIONS_LOG.md`
5. Create follow-up task to address properly

### Rule 3: Gate Failure Response
Failed quality gates must be addressed before proceeding.

**Process**:
1. Identify which gate failed
2. Understand root cause
3. Fix issue
4. Re-run gate check
5. Document resolution

## Workflow Rules

### Rule 1: Follow Defined Workflows
Use appropriate workflow from `workflows/` directory.

**Workflows**:
- Standard development cycle
- Feature request
- Bugfix
- Scientific model change
- Notebook to core promotion
- Release readiness
- Human-in-the-loop review

### Rule 2: Workflow Compliance
All steps in workflow must be completed.

**Enforcement**:
- Checklist for each workflow
- Status tracking in task board
- Artifact verification
- Sign-off requirements

### Rule 3: Workflow Adaptation
Workflow changes require Project Orchestrator approval.

**Process**:
1. Propose workflow change with rationale
2. Identify impact on existing processes
3. Get Orchestrator approval
4. Update workflow documentation
5. Notify all agents

## Communication Rules

### Rule 1: Clear Status Updates
Use standardized status terminology.

**Status values**:
- `not_started`: Task created but not begun
- `in_progress`: Actively working
- `blocked`: Waiting on dependency or decision
- `review`: Awaiting review
- `completed`: Finished and verified
- `cancelled`: No longer needed

### Rule 2: Blocker Escalation
Blockers must be escalated promptly.

**Process**:
1. Identify blocker
2. Update task status to "blocked"
3. Document blocker in task notes
4. Notify Project Orchestrator
5. Propose resolution if possible

### Rule 3: Decision Documentation
All significant decisions must be logged.

**Process**:
1. Document decision in `DECISIONS_LOG.md`
2. Include context, options considered, rationale
3. Record who made decision (agent or human)
4. Link to related tasks and artifacts

## Error Handling Rules

### Rule 1: Fail Gracefully
Errors should be caught and reported, not silently ignored.

**Implementation**:
- Use try-except blocks appropriately
- Log errors with context
- Provide actionable error messages
- Update task status to reflect error state

### Rule 2: Rollback on Failure
Failed changes should be reverted.

**Process**:
1. Detect failure (test failure, quality gate failure)
2. Revert code changes
3. Document failure in task notes
4. Analyze root cause
5. Propose fix
6. Retry with fix

### Rule 3: Learn from Failures
Failures should inform process improvements.

**Process**:
1. Document failure in `DECISIONS_LOG.md`
2. Identify process gap
3. Propose process improvement
4. Update protocols or workflows
5. Share learning with all agents

## Verification Rules

### Rule 1: Self-Verification
Agents must verify their own work before marking complete.

**Checklist**:
- [ ] Code runs without errors
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Quality gates passed
- [ ] Artifacts produced
- [ ] Status updated

### Rule 2: Peer Review
Significant changes require review by relevant specialist agent.

**Process**:
1. Complete work
2. Request review from appropriate agent
3. Address review comments
4. Get approval
5. Mark complete

### Rule 3: Human Verification
Critical changes require human verification.

**Triggers**:
- Scientific model changes
- Architectural changes
- Breaking changes
- Security-related changes
- Performance-critical changes

## State Management Rules

### Rule 1: Distinguish State Types
Clearly mark what is known vs. assumed vs. unknown.

**Categories**:
- **Verified**: Confirmed by tests or inspection
- **Assumed**: Believed true but not verified
- **Unknown**: Needs investigation

**Usage**: Always state which category applies when making claims.

### Rule 2: No Invented Names
Do not invent names for columns, functions, modules, or files.

**Process**:
1. Read actual code to find real names
2. Use exact names from codebase
3. If name doesn't exist, propose it explicitly
4. Get approval before using new names

### Rule 3: Version Awareness
Track which version of code/data/docs you're working with.

**Implementation**:
- Reference file paths explicitly
- Note last modification dates
- Check for recent changes before starting
- Update version info after changes

## Continuous Improvement

### Rule 1: Protocol Evolution
This protocol can be updated based on experience.

**Process**:
1. Identify protocol gap or improvement
2. Propose change to Project Orchestrator
3. Discuss with relevant agents
4. Get human approval for significant changes
5. Update protocol
6. Notify all agents

### Rule 2: Retrospectives
Regular review of what's working and what's not.

**Schedule**:
- After each major milestone
- Monthly for ongoing work
- After significant failures

**Output**: Updates to protocols, workflows, or agent definitions.

## Summary

These operating protocols ensure:
- ✅ Coordinated multi-agent development
- ✅ Scientific rigor and integrity
- ✅ Code quality and maintainability
- ✅ Human oversight of critical decisions
- ✅ Traceability and reproducibility
- ✅ Continuous improvement

**All agents must follow these protocols. When in doubt, ask for clarification.**

---

**Version**: 1.0  
**Last Updated**: 2026-05-07  
**Next Review**: After first major milestone
