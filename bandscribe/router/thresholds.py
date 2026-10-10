"""Router rule thresholds (DESIGN S4 table, the synthetic starting values; E6 re-fits them). A module of its own
so ``bandscribe.config`` can check ``[router.thresholds]`` keys without importing numpy / scipy.

``adt_over_r0`` is not in the DESIGN table (added 2026-10-10 before E6): the delayed peak must beat the zero-lag
correlation. A centred mono guitar has r0 = 1 and, being periodic, an off-lag peak of 0.5-0.75 at 9-36 ms too, so
without it every narrow window of a sustained part looked like ADT (Tier B smoke run, 5 of 14 centred windows)."""

THRESHOLDS: dict[str, float] = {
    "r0_sm_db": -12.0, "r0_coh": 0.95,
    "adt_xcorr": 0.5, "adt_min_lag_ms": 5.0, "adt_over_r0": 0.0,
    "r2_coh": 0.85, "r2_hard": 0.1, "r2_cen": 0.4,
    "r3_ild_db": 6.0, "r3_hard": 0.3, "r3_coh": 0.6,
}
