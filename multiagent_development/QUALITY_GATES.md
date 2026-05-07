# Quality Gates

Quality gates are mandatory checkpoints that all development activities must pass through. They ensure scientific rigor, code quality, and system stability.

## Gate Format
- Gate ID: GXXX
- Category: [Code|Scientific|Process|Data]
- Description: Gate requirements
- Validation: How the gate is checked
- Owner: Responsible agent
- Trigger: When the gate applies

## Active Quality Gates

### Code Quality Gates

#### G01 - No Unauthorized Modifications
**Category**: Code
**Description**: Existing code in src/, tests/, examples/, docs/, pyproject.toml, pytest.ini, or README.md cannot be modified without an approved change request
**Validation**: Pre-commit hooks check for unauthorized file modifications; change request log verification
**Owner**: Software Engineering Agent
**Trigger**: Any attempt to modify protected files

#### G02 - Test Coverage for Code Changes
**Category**: Code
**Description**: Every modification to src/ must include corresponding unit tests with adequate coverage
**Validation**: pytest coverage reports; automated CI checks
**Owner**: Testing QA Agent
**Trigger**: Pull request or commit to src/

#### G09 - Documentation Updates
**Category**: Code
**Description**: Documentation must be updated for every new feature, API change, or significant behavior modification
**Validation**: Documentation agent review; automated link checking
**Owner**: Documentation Agent
**Trigger**: Feature completion or API changes

### Scientific Quality Gates

#### G03 - Scientific Rationale Required
**Category**: Scientific
**Description**: Every scientific modification must include documented rationale, assumptions, and validation approach
**Validation**: Scientific review checklist; rationale documentation in commit messages
**Owner**: Scientific Architect
**Trigger**: Changes to physical models, algorithms, or methodological approaches

#### G05 - Baseline Inversion Isolation
**Category**: Scientific
**Description**: Baseline ERT inversion results must remain separate from the optimization loop and serve only as visual/geological reference
**Validation**: Code review to ensure inversion outputs don't feed into optimizer; test assertions
**Owner**: Geophysics ERT Agent
**Trigger**: Any changes to inversion or optimization integration

#### G06 - Data-Domain Optimization Only
**Category**: Scientific
**Description**: Optimization must operate exclusively in the data domain, comparing observed vs. simulated apparent resistivities
**Validation**: Code inspection; unit tests verifying optimization inputs/outputs
**Owner**: Optimizer & Tripwire Agent
**Trigger**: Optimizer implementation or modification

#### G07 - Tripwire Implementation Required
**Category**: Scientific
**Description**: All optimization runs must include tripwire checks for physical bounds violations, gradient stall, misfit stagnation, and numerical issues
**Validation**: Tripwire activation logs; test cases for tripwire conditions
**Owner**: Optimizer & Tripwire Agent
**Trigger**: Optimization execution or new optimizer features

#### G10 - Human Review for Conceptual Changes
**Category**: Scientific
**Description**: Any change to conceptual models (e.g., switching from Archie to Waxman-Smits) requires human review and approval
**Validation**: Human review queue processing; decision log documentation
**Owner**: Scientific Architect
**Trigger**: Proposals for fundamental model changes

### Data Quality Gates

#### G04 - Data Pipeline Preservation
**Category**: Data
**Description**: Modifications to data ingestion, processing, or QC pipelines must preserve all existing test cases and data validation rules
**Validation**: Full test suite execution; data integrity checks
**Owner**: Data Pipeline Guardian
**Trigger**: Changes to data processing components

### Process Quality Gates

#### G08 - Dashboard Advisory Only
**Category**: Process
**Description**: Diagnostic dashboard provides information and alerts but cannot make autonomous decisions or modify system state
**Validation**: Code review ensuring dashboard is read-only; UI/UX review
**Owner**: Dashboard HITL Agent
**Trigger**: Dashboard feature additions or modifications

## Gate Enforcement

### Automated Gates
- G01, G02, G04: Enforced through CI/CD pipelines and pre-commit hooks
- G05, G06, G07: Enforced through code review and automated tests

### Manual Gates
- G03, G08, G09, G10: Require agent review and approval

### Escalation
- Failed gates block integration until resolved
- Critical failures (G05, G06, G07 violations) trigger immediate human review
- Gate violations are logged in the decisions log

## Gate Evolution

Quality gates may be added, modified, or removed through the change request process. All gate changes require human review.

## Related Documents
- See `CHANGE_REQUESTS.md` for requesting gate modifications
- See `TASK_BOARD.md` for gate implementation tasks