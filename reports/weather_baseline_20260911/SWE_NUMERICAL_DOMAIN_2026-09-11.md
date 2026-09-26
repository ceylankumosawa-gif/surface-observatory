# SWE numerical-domain addendum before model outcomes

Streaming validation of the first completed standard-CDS snow files found a recurring tiny negative reference value: −7.345364920210326×10⁻²⁵ metres of water equivalent. It appears in nominally zero-snow grid cells in multiple pilot/date subsets. The same value is stored in the GRIB referenceValue header; it is not a new units conversion or a thermal-label-derived adjustment. Raw source values and packing headers are retained.

Extend the secondary protocol's already fixed numerical-domain tolerance to true GRIB141 SWE: only finite negative values in[-1e−12,0)metres are mapped tozero in the explicit merger. Preserve original raw SWE, a separate repair flag and counts, just as for soil moisture. Positive SWE is unchanged. Values below−1e−12 remain physically invalid and cannot enter the complete-native cohort. True missing values remain missing. This change is fixed before any new model fitting, outcomes or error inspection.

The version3 decoder preserves all decoded numerical SWE values, including negatives, instead of rejecting an entire requested file at its former nonnegative guard. It still verifies exact requested parameter/time identities, native cells, consolidated release, instantaneous step type, units, bitmap missingness and raw response hashes. Physical validity and the transparent zero repair are enforced by the separately versioned merger. Immutable source/request snapshots and all previous download/API charges survive continuation.

The model arms, rows before weather masking, folding rule, balancing recipe, hyperparameters, error gates and evaluation sequence remain unchanged. This does not repair the144 genuine source-native missing rows or reopen the stopped full-cohort primary.
