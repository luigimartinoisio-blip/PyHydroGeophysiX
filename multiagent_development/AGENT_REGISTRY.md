# Agent Registry

## Overview

This document provides a **quick reference** for all agents in the PyHydroGeophysiX_Carl multi-agent development system.

For detailed agent definitions including operational prompts, see individual files in `agents/` directory.

## Agent Summary Table

| ID | Agent Name | Primary Domain | Key Responsibilities |
|----|------------|----------------|---------------------|
| 00 | Project Orchestrator | Coordination | Task assignment, roadmap management, status tracking |
| 01 | Scientific Architect | Methodology | Scientific consistency, architecture design, assumption validation |
| 02 | Data Pipeline Guardian | Data Infrastructure | Data integrity, pipeline protection, QC validation |
| 03 | Hydrogeology Agent | Hydrology | Modflow integration, hydraulic states, saturation modeling |
| 04 | Geophysics ERT Agent | Geophysics | ERT processing, electrode geometry, apparent resistivity |
| 05 | Petrophysics Agent | Petrophysics | Van Genuchten-Archie, parameter bounds, identifiability |
| 06 | Forward Modeling Agent | Simulation | Forward chain, PyGIMLi integration, resistivity simulation |
| 07 | Optimizer & Tripwire Agent | Optimization | Misfit calculation, optimization, safety tripwires |
| 08 | Dashboard HITL Agent | Visualization | Diagnostic dashboard, alerts, human override interface |
| 09 | Software Engineering Agent | Code Quality | Architecture, refactoring, API design, maintainability |
| 10 | Testing QA Agent | Quality Assurance | Test coverage, fixtures, regression tests, validation |
| 11 | Documentation Agent | Documentation | API docs, guides, changelog, scientific documentation |

## Agent Interaction Matrix

Shows which agents commonly interact:

```
        00  01  02  03  04  05  06  07  08  09  10  11
00 Orch  -   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●
01 Sci   ●   -   ●   ●   ●   ●   ●   ●   ●   ○   ○   ●
02 Data  ●   ●   -   ●   ●   ○   ○   ○   ○   ○   ●   ○
03 Hydro ●   ●   ●   -   ○   ●   ●   ○   ○   ○   ●   ○
04 ERT   ●   ●   ●   ○   -   ○   ●   ○   ●   ○   ●   ○
05 Petro ●   ●   ○   ●   ○   -   ●   ●   ○   ○   ●   ○
06 Fwd   ●   ●   ○   ●   ●   ●   -   ●   ○   ○   ●   ○
07 Opt   ●   ●   ○   ○   ○   ●   ●   -   ●   ○   ●   ○
08 Dash  ●   ●   ○   ○   ●   ○   ○   ●   -   ●   ●   ○
09 SWE   ●   ○   ○   ○   ○   ○   ○   ○   ●   -   ●   ●
10 Test  ●   ○   ●   ●   ●   ●   ●   ●   ●   ●   -   ○
11 Doc   ●   ●   ○   ○   ○   ○   ○   ○   ○   ●   ○   -

Legend: ● = Frequent interaction, ○ = Occasional interaction, - = Self
```

## Agent Hierarchy

```
Project Orchestrator (00)
├── Scientific Architect (01)
│   ├── Hydrogeology Agent (03)
│   ├── Geophysics ERT Agent (04)
│   ├── Petrophysics Agent (05)
│   ├── Forward Modeling Agent (06)
│   └── Optimizer & Tripwire Agent (07)
├── Data Pipeline Guardian (02)
├── Dashboard HITL Agent (08)
├── Software Engineering Agent (09)
├── Testing QA Agent (10)
└── Documentation Agent (11)
```

## Agent Descriptions

### 00 - Project Orchestrator

**Role**: Overall coordination and project management

**Key Responsibilities**:
- Maintain development roadmap
- Assign tasks to appropriate agents
- Track progress and status
- Manage dependencies between tasks
- Escalate blockers
- Coordinate releases
- Maintain task board
- Generate status reports

**Authority Level**: High - Can assign tasks and coordinate all agents

**Human Interaction**: Regular - Reports status and escalates critical decisions

**File**: `agents/00_project_orchestrator.md`

---

### 01 - Scientific Architect

**Role**: Guardian of scientific methodology and architectural consistency

**Key Responsibilities**:
- Ensure methodological consistency
- Validate scientific assumptions
- Review architectural decisions
- Approve scientific model changes
- Maintain conceptual architecture
- Validate data-domain optimization approach
- Ensure baseline inversion isolation
- Review tripwire logic

**Authority Level**: High - Can veto scientifically unsound changes

**Human Interaction**: Frequent - Escalates conceptual decisions

**File**: `agents/01_scientific_architect.md`

---

### 02 - Data Pipeline Guardian

**Role**: Protector of data infrastructure and quality

**Key Responsibilities**:
- Protect data ingestion pipeline
- Validate data connectors
- Ensure temporal alignment
- Maintain quality control checks
- Review data transformations
- Validate data contracts
- Prevent breaking changes to data flow
- Monitor data integrity

**Authority Level**: High - Can block changes that threaten data integrity

**Human Interaction**: Moderate - Escalates data contract changes

**File**: `agents/02_data_pipeline_guardian.md`

---

### 03 - Hydrogeology Agent

**Role**: Specialist in hydrogeological modeling

**Key Responsibilities**:
- Modflow integration and configuration
- Hydraulic head calculations
- Saturation state computation
- Temporal alignment with ERT observations
- Van Genuchten parameter handling
- Groundwater flow validation
- Boundary condition management
- Hydrogeological assumption documentation

**Authority Level**: Medium - Expert in hydrology domain

**Human Interaction**: Moderate - Escalates hydrogeological model changes

**File**: `agents/03_hydrogeology_agent.md`

---

### 04 - Geophysics ERT Agent

**Role**: Specialist in electrical resistivity tomography

**Key Responsibilities**:
- ERT data processing
- Electrode geometry management
- Apparent resistivity handling
- Baseline inversion (visual reference only)
- ERT measurement quality control
- Geometric factor calculations
- ERT-specific assumptions
- Geophysical validation

**Authority Level**: Medium - Expert in ERT domain

**Human Interaction**: Moderate - Escalates geophysical model changes

**File**: `agents/04_geophysics_ert_agent.md`

---

### 05 - Petrophysics Agent

**Role**: Specialist in petrophysical relationships

**Key Responsibilities**:
- Van Genuchten-Archie relationship
- Petrophysical parameter bounds
- Parameter identifiability analysis
- Alternative models (Waxman-Smits) evaluation
- Cementation exponent (m) handling
- Saturation exponent (n) handling
- Formation factor calculations
- Petrophysical assumption validation

**Authority Level**: Medium - Expert in petrophysics domain

**Human Interaction**: Frequent - Escalates petrophysical model changes

**File**: `agents/05_petrophysics_agent.md`

---

### 06 - Forward Modeling Agent

**Role**: Specialist in forward simulation chain

**Key Responsibilities**:
- Hydro-to-resistivity forward chain
- PyGIMLi integration
- True resistivity field generation
- Apparent resistivity simulation
- Forward modeling validation
- Mesh handling
- Numerical accuracy
- Forward model documentation

**Authority Level**: Medium - Expert in forward modeling

**Human Interaction**: Moderate - Escalates forward model changes

**File**: `agents/06_forward_modeling_agent.md`

---

### 07 - Optimizer & Tripwire Agent

**Role**: Specialist in optimization and safety monitoring

**Key Responsibilities**:
- Data-domain misfit calculation
- Optimization algorithm implementation
- Gradient computation and validation
- Tripwire implementation (bounds, stall, identifiability)
- Convergence monitoring
- Parameter update logic
- Safety check enforcement
- Optimization diagnostics

**Authority Level**: High - Can halt optimization on tripwire activation

**Human Interaction**: Frequent - Escalates tripwire activations

**File**: `agents/07_optimizer_tripwire_agent.md`

---

### 08 - Dashboard HITL Agent

**Role**: Specialist in human-in-the-loop interface

**Key Responsibilities**:
- Diagnostic dashboard design
- Residual heatmap visualization
- Jacobian sensitivity visualization
- Tripwire alert display
- Human override mechanism
- Decision logging interface
- Dashboard technology selection
- User experience design

**Authority Level**: Medium - Designs interface, doesn't make decisions

**Human Interaction**: Frequent - Designs for human interaction

**File**: `agents/08_dashboard_hitl_agent.md`

---

### 09 - Software Engineering Agent

**Role**: Guardian of code quality and architecture

**Key Responsibilities**:
- Code architecture review
- Refactoring proposals
- API design
- Module organization
- Code style consistency
- Performance optimization
- Dependency management
- Maintainability assessment

**Authority Level**: Medium - Can request refactoring

**Human Interaction**: Low - Escalates architectural changes

**File**: `agents/09_software_engineering_agent.md`

---

### 10 - Testing QA Agent

**Role**: Guardian of test coverage and quality

**Key Responsibilities**:
- Test suite maintenance
- Unit test creation
- Integration test creation
- Test fixture design
- Regression test implementation
- Test coverage monitoring
- Quality gate enforcement
- Validation strategy

**Authority Level**: High - Can block integration on test failures

**Human Interaction**: Low - Escalates persistent test failures

**File**: `agents/10_testing_qa_agent.md`

---

### 11 - Documentation Agent

**Role**: Guardian of documentation quality

**Key Responsibilities**:
- API documentation
- User guide maintenance
- Developer guide maintenance
- Scientific methodology documentation
- Changelog maintenance
- Example notebook documentation
- README updates
- Documentation consistency

**Authority Level**: Medium - Can request documentation

**Human Interaction**: Low - Escalates documentation gaps

**File**: `agents/11_documentation_agent.md`

## Agent Activation Triggers

### When to Activate Each Agent

**Project Orchestrator (00)**:
- Start of new development cycle
- Task assignment needed
- Status reporting required
- Blocker escalation
- Release planning

**Scientific Architect (01)**:
- Scientific model change proposed
- Architectural decision needed
- Assumption validation required
- Conceptual inconsistency detected
- New scientific feature proposed

**Data Pipeline Guardian (02)**:
- Data ingestion change proposed
- Data contract modification needed
- QC logic change proposed
- Temporal alignment issue
- Data integrity concern

**Hydrogeology Agent (03)**:
- Modflow integration change
- Hydraulic state calculation change
- Van Genuchten parameter change
- Saturation computation change
- Hydrogeological assumption change

**Geophysics ERT Agent (04)**:
- ERT processing change
- Electrode geometry change
- Apparent resistivity handling change
- Baseline inversion change
- ERT QC change

**Petrophysics Agent (05)**:
- Archie relationship change
- Petrophysical parameter change
- Parameter bounds change
- Alternative model evaluation
- Identifiability analysis needed

**Forward Modeling Agent (06)**:
- Forward chain modification
- PyGIMLi integration change
- Resistivity simulation change
- Mesh handling change
- Forward model validation needed

**Optimizer & Tripwire Agent (07)**:
- Optimization algorithm change
- Misfit function change
- Tripwire logic change
- Convergence criteria change
- Safety check modification

**Dashboard HITL Agent (08)**:
- Dashboard feature addition
- Visualization change
- Alert mechanism change
- Override interface change
- User experience improvement

**Software Engineering Agent (09)**:
- Code refactoring proposed
- API design change
- Module reorganization
- Performance optimization
- Architecture review needed

**Testing QA Agent (10)**:
- New feature needs tests
- Test failure investigation
- Test coverage gap identified
- Regression test needed
- Quality gate evaluation

**Documentation Agent (11)**:
- New feature needs documentation
- API change needs documentation
- User guide update needed
- Scientific documentation needed
- Changelog update required

## Agent Communication Channels

### Primary Communication Files

**Task Board**: `TASK_BOARD.md`
- All agents read for task assignments
- Assigned agent updates status

**Change Requests**: `CHANGE_REQUESTS.md`
- Agents propose changes
- Orchestrator reviews and assigns

**Human Review Queue**: `HUMAN_REVIEW_QUEUE.md`
- Agents add items needing human decision
- Humans provide decisions

**Decisions Log**: `DECISIONS_LOG.md`
- All agents record significant decisions
- Reference for future work

**Reports**: `reports/` directory
- Agents generate status reports
- Orchestrator consolidates

## Agent Collaboration Patterns

### Pattern 1: Sequential Handoff
One agent completes work, hands off to next agent.

**Example**: Feature Request → Scientific Architect → Implementation Agent → Testing QA → Documentation

### Pattern 2: Parallel Review
Multiple agents review simultaneously.

**Example**: Code change → Software Engineering + Testing QA + Documentation (parallel)

### Pattern 3: Escalation Chain
Agent escalates to higher authority.

**Example**: Implementation issue → Domain Agent → Scientific Architect → Human Review

### Pattern 4: Collaborative Design
Multiple agents contribute to design.

**Example**: New feature design → Scientific Architect + Domain Agents + Software Engineering

## Version History

- **v1.0** (2026-05-07): Initial agent registry with 12 agents

---

**For detailed agent definitions, see individual files in `agents/` directory.**
