# UZF Unsaturated-Zone Exploratory Assessment

This directory contains the exploratory multi-agent assessment of the MODFLOW Unsaturated-Zone Flow package (UZF) for the PyHydroGeophysiX_Carl project.

## Final Status

**STRATEGICALLY_RELEVANT_BUT_PARKED**

UZF is considered scientifically and strategically relevant for possible future unsaturated-zone modeling, but it is not part of the current MVP.

## Closure Status

**CLOSED_AS_STRATEGIC_ASSESSMENT**

The current exploratory task is considered closed as a strategic scientific assessment.

## Implementation Authorization

**NOT_AUTHORIZED**

No implementation is authorized at this stage.

This means:

- do not modify `src/`;
- do not modify `tests/`;
- do not modify `examples/`;
- do not modify `docs/`;
- do not modify the current `HydroState` data contract;
- do not add UZF dependencies;
- do not run UZF/MODFLOW sandbox tests yet;
- do not include UZF in the current MVP.

## Summary

The MODFLOW UZF package was reviewed as a possible future component for representing unsaturated-zone flow in PyHydroGeophysiX_Carl.

The assessment concluded that UZF may become useful later for modeling vadose-zone dynamics, moisture redistribution, recharge processes, and possible future petrophysical coupling. However, the package is not currently needed for the MVP and should not be integrated until several uncertainties are resolved.

The main reason for parking UZF is that the current MVP should remain focused on:

- validated data ingestion;
- ERT connectors;
- temporal and spatial alignment;
- basic/saturated hydrogeological modeling;
- petrophysical bridge development;
- data-domain optimization;
- tripwire logic;
- human-in-the-loop supervision.

Adding UZF too early would increase scientific and software complexity before the core workflow is validated.

## Main Reasons for Parking UZF

UZF remains parked because of the following unresolved issues:

1. **Unverified output variables**  
   It is not yet confirmed whether the specific UZF implementation provides direct access to variables such as moisture content, saturation, capillary pressure, recharge, evapotranspiration, or vertical flux.

2. **Parameter identifiability risk**  
   UZF may require several hydraulic parameters, such as saturated hydraulic conductivity and van Genuchten parameters. These may be difficult to estimate reliably at field scale.

3. **Equifinality risk**  
   UZF parameters may trade off with Archie/petrophysical parameters and ERT inversion uncertainty. This could make the coupled inverse problem underdetermined.

4. **Physical approximation limitations**  
   UZF relies on simplified unsaturated-flow assumptions, including a kinematic-wave approximation. This may be inappropriate for fine-grained soils, capillary-dominated flow, perched water, hysteresis, or strong lateral unsaturated flow.

5. **MVP scope protection**  
   UZF is not required for the current MVP and should not distract from validating the basic hydrogeophysical data-assimilation loop.

## Reports Included in This Directory

This directory should contain:

- `hydro_unsaturated_package_assessment.md`  
  Hydrogeology Agent assessment of the MODFLOW UZF package, including the R2 minor revision after scientific review.

- `hydro_unsaturated_package_scientific_review.md`  
  Scientific Architect review of the Hydrogeology Agent assessment.

Optional future files may include:

- `human_reading_notes.md`  
  Notes written by the human project owner while personally studying the UZF documentation.

- `uzf_output_variable_verification.md`  
  Future Phase B0 verification of UZF output variables.

## Current Decision

UZF is classified as:

**STRATEGICALLY_RELEVANT_BUT_PARKED**

This means:

- UZF is interesting for the future;
- UZF should remain documented;
- UZF should not be forgotten;
- UZF should not be implemented now;
- UZF should not affect the current MVP;
- UZF can only be reconsidered after a specific verification phase.

## Future Unlock Condition

UZF may only move from parked status to active investigation after a future **Phase B0: Output-variable verification** task.

That future task must verify:

- the exact MODFLOW version and UZF variant;
- whether moisture content `θ` is available as a direct output, internal state, or derived quantity;
- whether saturation `S` is available as a direct output, internal state, or derived quantity;
- whether pressure head or capillary pressure `ψ` is available;
- whether recharge is available;
- whether evapotranspiration is available;
- whether vertical flux is available;
- whether these variables are accessible through the intended MODFLOW/FloPy workflow.

No sandbox testing, HydroState extension, dependency addition, or `src/` modification is authorized before this verification and human approval.

## Human Questions Still Open

The following questions remain open for future work:

1. Is explicit unsaturated-zone modeling required after the MVP?
2. Which MODFLOW/UZF variant should PyHydroGeophysiX_Carl target?
3. Are field-scale soil hydraulic parameters available for the intended field sites?
4. Is ERT expected to resolve vadose-zone moisture changes at the target scale?
5. Which hydrological state variable is strictly required by the first petrophysical bridge?
6. Should UZF remain a future optional module or become part of a later development phase?

## Recommended Next Action

For now, no further software action is required.

Recommended immediate action:

1. Archive the two UZF reports in this directory.
2. Keep UZF outside the MVP.
3. Continue development of the core PyHydroGeophysiX_Carl workflow.
4. If the human owner studies the UZF documentation, store notes in `human_reading_notes.md`.
5. Reopen the topic only through a future Phase B0 output-variable verification task.

## Final Statement

The UZF exploratory assessment is closed as a strategic scientific note.

Final status:

**STRATEGICALLY_RELEVANT_BUT_PARKED**

Implementation status:

**NOT_AUTHORIZED**