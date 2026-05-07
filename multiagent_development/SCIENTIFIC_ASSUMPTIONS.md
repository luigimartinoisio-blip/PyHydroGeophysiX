# Scientific Assumptions

This document explicitly lists all scientific assumptions underlying PyHydroGeophysiX_Carl, distinguishing between verified assumptions (based on current implementation and testing) and assumptions to be confirmed (requiring further validation or human review).

## Verified Assumptions ✅

### Petrophysical Relationships
- **Archie Law**: Electrical conductivity is related to porosity, saturation, and cementation factor (verified in current petrophysical bridge implementation)
- **Van Genuchten Model**: Soil water retention curve follows Van Genuchten parameterization (verified in hydrological modeling)

### Geophysical Forward Modeling
- **ERT Forward Modeling**: PyGIMLi provides accurate forward modeling of electrical resistivity tomography (verified through existing notebooks and tests)
- **Apparent Resistivity Calculation**: Forward modeling produces apparent resistivity values comparable to observed data (verified in examples)

### Hydrogeological Modeling
- **MODFLOW Compatibility**: Hydrological states from MODFLOW can be integrated with geophysical data (verified in coupled modeling examples)
- **Temporal Alignment**: Time-series data from hydrological and geophysical sources can be properly synchronized (verified in data processing)

### Data Processing
- **Baseline ERT Inversion**: Serves as visual/geological reference and does not participate in optimization loop (verified by design)
- **Data Quality Control**: Implemented QC procedures adequately filter invalid measurements (verified in data pipeline)

## Assumptions to Confirm ❓

### Optimization Domain
- **Data-Domain Optimization**: Optimization should compare observed vs. simulated apparent resistivities rather than inverted models (requires validation of convergence and accuracy)
- **Parameter Identifiability**: All optimized petrophysical parameters are identifiable from ERT data (requires sensitivity analysis)

### Tripwire Conditions
- **Physical Bounds Enforcement**: Defined parameter bounds adequately represent physical constraints (requires expert review)
- **Gradient Stall Detection**: Current gradient-based stall detection is appropriate for this optimization problem (requires testing)
- **Misfit Stagnation Criteria**: Defined stagnation thresholds prevent false positives while catching real issues (requires benchmarking)

### Model Extensions
- **Waxman-Smits Alternative**: Waxman-Smits model provides better petrophysical relationships for clay-rich soils (requires comparative study)
- **Advanced Forward Modeling**: Current forward chain adequately represents the physics (requires validation against analytical solutions)

### Data Assimilation Architecture
- **Human-in-the-Loop Integration**: Proposed HITL architecture improves convergence over fully automated approaches (requires user studies)
- **Diagnostic Dashboard Effectiveness**: Visual diagnostics help identify optimization issues (requires usability testing)

## Assumption Validation Process

When confirming assumptions:
1. Design validation experiment or analysis
2. Execute using existing notebooks or new test cases
3. Document results in experiment logs
4. Update this document with confirmation status
5. If assumption invalidated, initiate change request for modification

## Related Documents
- See `HUMAN_REVIEW_QUEUE.md` for assumptions requiring immediate human confirmation
- See `RISK_REGISTER.md` for risks associated with unconfirmed assumptions