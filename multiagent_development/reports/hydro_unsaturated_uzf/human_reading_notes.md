

**MODFLOW-2005 (Old approach):**
Recharge is applied instantaneously to the water table. It lacks a physical representation of the unsaturated zone, meaning there is no time lag, no intermediate storage, and no distinction between infiltration and actual recharge.

**UZF1 Package approach:**
Water infiltrates at the land surface and travels through the unsaturated zone using the kinematic wave approximation and the Brooks-Corey function. This simulates:
*   **A) Time Lag:** The delay between surface infiltration and actual groundwater recharge.
*   **B) Storage:** The capacity of the soil to retain water, preventing recharge until a moisture threshold is met.
*   **C) Flow Partitioning:** The division of surface water into infiltration, runoff, and evapotranspiration based on soil suction and saturation.

UZF1 Advanced Processes:

- Infiltration Limit: The infiltration rate is capped by the saturated vertical hydraulic conductivity (Kvs). Any excess water that cannot enter the soil is treated as surface runoff.
- Two-Stage Evapotranspiration (ET): Unlike the standard ET package, UZF1 first extracts water from the unsaturated zone storage. If the ET demand is still not met, it then removes water directly from the groundwater, provided the water table is within the extinction depth.
- Surface Discharge & Routing: If the water table rises above the land surface, water is discharged. This discharge, along with excess infiltration (runoff), can be dynamically routed as inflow to streams (SFR package) or lakes (LAK package). Without these packages, the water is permanently removed from the simulation.
In summary: UZF1 creates a coupled system where soil capacity limits input, ET prioritizes soil moisture over groundwater, and excess water is converted into surface flow rather than being ignored.