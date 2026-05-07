# Human Review Queue

This document lists decisions that require human expertise and cannot be made autonomously by agents.

## Review Format
- Decision ID: HRXXX
- Category: [Scientific|Technical|Architectural]
- Description: Decision required
- Context: Background information
- Options: Available choices
- Urgency: [Low|Medium|High]
- Requested By: Agent requesting review
- Status: [Pending|In Review|Approved|Rejected]

## Pending Human Reviews

### Scientific Decisions

#### HR001 - Initial Petrophysical Bounds
**Category**: Scientific
**Description**: Confirm the initial physical bounds for petrophysical parameters (porosity, cementation factor, saturation exponents)
**Context**: Current bounds are based on literature values but need validation against expected field conditions
**Options**: 
- Approve current bounds as reasonable starting point
- Adjust bounds based on specific site characteristics
- Require sensitivity analysis before setting bounds
**Urgency**: High
**Requested By**: Petrophysics Agent
**Status**: Pending

#### HR002 - ERT Observed Data Schema
**Category**: Scientific
**Description**: Confirm the final data schema for observed ERT apparent resistivity measurements
**Context**: Schema must support both synthetic and real field data with proper metadata
**Options**:
- Standard PyGIMLi format with extensions
- Custom schema optimized for data assimilation
- Hybrid approach with conversion utilities
**Urgency**: High
**Requested By**: Geophysics ERT Agent
**Status**: Pending

#### HR003 - HydroState Schema Definition
**Category**: Scientific
**Description**: Confirm the schema for hydrological state variables (heads, saturations, fluxes)
**Context**: Must be compatible with MODFLOW outputs and forward modeling inputs
**Options**:
- Minimal schema with essential variables
- Extended schema with derived quantities
- Modular schema supporting multiple hydrological models
**Urgency**: High
**Requested By**: Hydrogeology Agent
**Status**: Pending

#### HR004 - Optimizable Parameters for MVP
**Category**: Scientific
**Description**: Confirm which parameters should be optimizable in the first minimum viable product
**Context**: Balance between identifiability, computational cost, and scientific value
**Options**:
- Core Archie parameters only (porosity, cementation, saturation)
- Extended set including Van Genuchten parameters
- Phased approach starting with subset
**Urgency**: Medium
**Requested By**: Scientific Architect
**Status**: Pending

#### HR005 - Waxman-Smits as Future Option
**Category**: Scientific
**Description**: Confirm that Waxman-Smits model should remain as future enhancement rather than initial implementation
**Context**: Archie law is simpler and sufficient for initial validation
**Options**:
- Keep as future option (recommended)
- Implement both models from start
- Replace Archie with Waxman-Smits
**Urgency**: Low
**Requested By**: Petrophysics Agent
**Status**: Pending

### Technical Decisions

#### HR006 - Dashboard Technology Stack
**Category**: Technical
**Description**: Confirm the technology choice for the Human-in-the-Loop diagnostic dashboard
**Context**: Must support real-time visualization, interactive controls, and integration with Python backend
**Options**:
- Streamlit for rapid prototyping
- Dash/Plotly for advanced interactivity
- Custom web application with Flask/React
**Urgency**: Medium
**Requested By**: Dashboard HITL Agent
**Status**: Pending

#### HR007 - Testing Strategy for Synthetic vs Real Data
**Category**: Technical
**Description**: Confirm the strategy for testing with synthetic benchmarks vs real field data
**Context**: Need to ensure robustness across different data quality levels
**Options**:
- Primary testing with synthetic data, validation with real data
- Parallel testing with both data types
- Phased approach: synthetic first, then real data
**Urgency**: Medium
**Requested By**: Testing QA Agent
**Status**: Pending

## Review Process

1. **Submission**: Agent identifies need for human decision and adds to this queue
2. **Prioritization**: Project Orchestrator reviews and prioritizes based on urgency and dependencies
3. **Review**: Human expert reviews options and provides decision
4. **Documentation**: Decision recorded in DECISIONS_LOG.md
5. **Implementation**: Approved decisions implemented by responsible agents
6. **Closure**: Item removed from queue upon implementation

## Current Queue Status
- Total Pending: 7
- High Priority: 3
- Medium Priority: 3
- Low Priority: 1

## Related Documents
- See `DECISIONS_LOG.md` for completed reviews
- See `TASK_BOARD.md` for implementation tasks dependent on these decisions