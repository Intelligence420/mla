"""tool_pipeline.schema — RunConfig (inputs) and RunResult (outputs).

RunConfig : operand subscripts, output indices, dim sizes, dtype + accumulate
            dtype, tile (TM,TN,TK), swizzle flag, which baselines to run.
RunResult : status, accuracy (max/mean/rel err), metrics (ms distribution,
            TFLOP/s, GB/s, arithmetic intensity, %-of-peak), compile-vs-run
            time, provenance (GPU clock/temp/power), path to persisted kernel.

The contract between core and GUI — define FIRST (ticket T0.2).
"""
