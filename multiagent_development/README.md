# PyHydroGeophysiX_Carl Multi-Agent Development System

## Overview

This directory contains the **Multi-Agent Development Control Room** for PyHydroGeophysiX_Carl.

**IMPORTANT**: This system is **additive** and **non-invasive**. It does not replace or modify the existing project structure, tests, or workflows. Instead, it provides a coordination layer for AI agents to collaboratively evolve the project while maintaining scientific rigor and software quality.

## Purpose

The multi-agent system serves as a control room for:

- **Planning**: Roadmap management and task prioritization
- **Coordination**: Agent assignment and workflow orchestration
- **Scientific Review**: Ensuring methodological consistency
- **Code Evolution**: Controlled enhancement of the codebase
- **Quality Control**: Testing, validation, and quality gates
- **Risk Management**: Tracking and mitigating scientific and technical risks
- **Documentation**: Maintaining comprehensive project documentation
- **Human-in-the-Loop**: Managing decisions requiring human expertise

## Directory Structure

```
multiagent_development/
├── README.md                              # This file
├── DEVELOPMENT_MISSION.md                 # Core mission and objectives
├── MULTIAGENT_OPERATING_PROTOCOL.md       # Operating rules and principles
├── AGENT_REGISTRY.md                      # Agent roles and responsibilities
├── DEVELOPMENT_ROADMAP.md                 # Phased development plan
├── TASK_BOARD.md                          # Active and planned tasks
├── DECISIONS_LOG.md                       # Record of key decisions
├── SCIENTIFIC_ASSUMPTIONS.md              # Explicit scientific assumptions
├── RISK_REGISTER.md                       # Risk tracking and mitigation
├── QUALITY_GATES.md                       # Quality control checkpoints
├── HUMAN_REVIEW_QUEUE.md                  # Decisions awaiting human input
├── CHANGE_REQUESTS.md                     # Change request registry
├── agents/                                # Agent definitions (12 agents)
├── workflows/                             # Development workflows (7 workflows)
├── templates/                             # Operational templates (7 templates)
└── reports/                               # Status and progress reports (5 reports)
```

## Key Principles

### 1. Non-Invasive Operation
- The existing project structure remains unchanged
- All modifications go through formal change requests
- Tested workflows are protected

### 2. Scientific Rigor
- Methodological changes require human review
- Assumptions are explicitly documented
- Physical constraints are enforced

### 3. Quality First
- Every code change requires tests
- Quality gates must pass before integration
- Documentation is mandatory

### 4. Human-in-the-Loop
- Critical decisions require human approval
- Tripwire alerts trigger human review
- Override mechanisms are always available

## Agent System

The system consists of **12 specialized agents**:

1. **Project Orchestrator** - Overall coordination
2. **Scientific Architect** - Methodological consistency
3. **Data Pipeline Guardian** - Data integrity protection
4. **Hydrogeology Agent** - Hydrological modeling
5. **Geophysics ERT Agent** - Electrical resistivity tomography
6. **Petrophysics Agent** - Petrophysical relationships
7. **Forward Modeling Agent** - Forward simulation chain
8. **Optimizer & Tripwire Agent** - Optimization and safety
9. **Dashboard HITL Agent** - Human-in-the-loop interface
10. **Software Engineering Agent** - Code quality and architecture
11. **Testing QA Agent** - Test coverage and validation
12. **Documentation Agent** - Documentation maintenance

See `AGENT_REGISTRY.md` for detailed agent descriptions.

## Workflow System

The system defines **7 core workflows**:

1. **Standard Development Cycle** - Normal development process
2. **Feature Request Workflow** - Adding new capabilities
3. **Bugfix Workflow** - Addressing defects
4. **Scientific Model Change Workflow** - Modifying physical models
5. **Notebook to Core Promotion Workflow** - Promoting validated code
6. **Release Readiness Workflow** - Preparing releases
7. **Human-in-the-Loop Review Workflow** - Human decision points

See `workflows/` directory for detailed workflow descriptions.

## Getting Started

### For AI Agents

1. Read `MULTIAGENT_OPERATING_PROTOCOL.md` for operating rules
2. Review `AGENT_REGISTRY.md` to understand your role
3. Check `TASK_BOARD.md` for assigned tasks
4. Follow the appropriate workflow from `workflows/`
5. Use templates from `templates/` for consistency
6. Update reports in `reports/` after completing tasks

### For Human Developers

1. Review `DEVELOPMENT_MISSION.md` for project vision
2. Check `HUMAN_REVIEW_QUEUE.md` for pending decisions
3. Review `TASK_BOARD.md` for development status
4. Consult `RISK_REGISTER.md` for known risks
5. Use `CHANGE_REQUESTS.md` to propose modifications
6. Follow `QUALITY_GATES.md` for quality standards

## Scientific Context

PyHydroGeophysiX_Carl is a hydro-geophysical framework targeting **Data Assimilation with Human-in-the-Loop**.

### Conceptual Architecture (Future Vision)

1. **Data Infrastructure & Quality Control**
2. **ERT Geophysical Processing**
3. **Hydrogeological Modeling**
4. **Petrophysical Bridge**
5. **Geophysical Forward Modeling**
6. **Data-Domain Optimizer**
7. **Tripwire Layer**
8. **Diagnostic Dashboard**
9. **Human-in-the-Loop Review**

### Key Methodological Principles

1. **Baseline ERT inversion** is a visual/geological reference, NOT part of the optimization loop
2. **Optimization operates in data domain**: comparing observed vs. simulated apparent resistivities
3. **Tripwire system** monitors physical bounds, gradient stall, misfit stagnation, parameter identifiability
4. **Human review required** for conceptual model changes (e.g., Archie to Waxman-Smits)
5. **No autonomous modifications** to validated scientific code without formal approval

## Current Project State

The project has been:
- ✅ Restructured with clean `src/` organization
- ✅ Tested with initial test suite
- ✅ Documented with architecture and API docs
- ✅ Organized with example notebooks and scripts

**This multi-agent system builds on top of this stable foundation.**

## Quality Gates

All development must pass through defined quality gates:

- **G01**: No modifications without change request
- **G02**: All code changes require tests
- **G03**: Scientific changes require rationale
- **G04**: Data pipeline changes preserve existing tests
- **G05**: Baseline inversion stays out of optimization loop
- **G06**: Optimization only in data domain
- **G07**: Tripwire required for safety checks
- **G08**: Dashboard is advisory, not autonomous
- **G09**: Documentation updated with features
- **G10**: Human review for conceptual changes

See `QUALITY_GATES.md` for complete details.

## Contact and Support

For questions about:
- **Scientific methodology**: Consult Scientific Architect agent
- **Code architecture**: Consult Software Engineering agent
- **Testing strategy**: Consult Testing QA agent
- **Documentation**: Consult Documentation agent
- **Overall coordination**: Consult Project Orchestrator agent

## Version History

- **v0.1.0** (2026-05-07): Initial multi-agent system creation
  - 12 agents defined
  - 7 workflows established
  - Quality gates implemented
  - Initial task board created

---

**Remember**: This system exists to help the project evolve safely and scientifically. When in doubt, ask for human review.
