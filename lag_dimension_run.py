from __future__ import annotations

import os
import sys


def _option_or_environment(name: str, environment_name: str, default: int) -> int:
    prefix = f"{name}="
    for index, argument in enumerate(sys.argv):
        if argument.startswith(prefix):
            return int(argument[len(prefix) :])
        if argument == name and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
    return int(os.environ.get(environment_name, default))


output_lags = _option_or_environment(
    "--output-lags", "FA_SAL_OUTPUT_LAGS", 2
)
input_lags = _option_or_environment(
    "--input-lags", "FA_SAL_INPUT_LAGS", 1
)
os.environ["FA_SAL_OUTPUT_LAGS"] = str(output_lags)
os.environ["FA_SAL_INPUT_LAGS"] = str(input_lags)

import lag_dimension_sweep
import lag_dimension_experiment as lagged

lagged.OUTPUT_LAGS = int(os.environ["FA_SAL_OUTPUT_LAGS"])
lagged.INPUT_LAGS = int(os.environ["FA_SAL_INPUT_LAGS"])


if __name__ == "__main__":
    import main

    main.main()
