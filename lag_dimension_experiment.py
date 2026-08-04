
from __future__ import annotations

import sys
from typing import Dict, Tuple

import numpy as np

import environment
import experiment


def _consume_int_option(name: str, default: int) -> int:
    prefix = f"{name}="
    for index, argument in enumerate(list(sys.argv)):
        if argument.startswith(prefix):
            value = int(argument[len(prefix) :])
            del sys.argv[index]
            return value
        if argument == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires an integer value")
            value = int(sys.argv[index + 1])
            del sys.argv[index : index + 2]
            return value
    return int(default)


OUTPUT_LAGS = _consume_int_option("--output-lags", 2)
INPUT_LAGS = _consume_int_option("--input-lags", 1)
if OUTPUT_LAGS < 2:
    raise SystemExit("--output-lags must be at least 2")
if INPUT_LAGS < 1:
    raise SystemExit("--input-lags must be at least 1")


class LaggedNARXDoubleIntegratorEnv(environment.NARXDoubleIntegratorEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_lags = int(OUTPUT_LAGS)
        self.input_lags = int(INPUT_LAGS)
        self._observed_histories: Dict[
            Tuple[float, float], Tuple[np.ndarray, np.ndarray]
        ] = {}

    @staticmethod
    def _key(y_t: float, y_tm1: float) -> Tuple[float, float]:
        return (round(float(y_t), 12), round(float(y_tm1), 12))

    @property
    def input_dimension(self) -> int:
        return int(self.output_lags + self.input_lags)

    def _base_regressor(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.asarray(
            [z[0], z[1], z[self.output_lags]],
            dtype=float,
        )

    def _default_outputs(self, y_t: float, y_tm1: float) -> np.ndarray:
        outputs = [float(y_t), float(y_tm1)]
        velocity = float(y_t) - float(y_tm1)
        while len(outputs) < self.output_lags:
            outputs.append(outputs[-1] - velocity)
        return np.asarray(outputs, dtype=float)

    def make_regressor(self, y_t: float, y_tm1: float, u_t: float) -> np.ndarray:
        if self.output_lags == 2 and self.input_lags == 1:
            return super().make_regressor(y_t, y_tm1, u_t)

        history = self._observed_histories.get(self._key(y_t, y_tm1))
        if history is None:
            outputs = self._default_outputs(y_t, y_tm1)
            past_controls = np.zeros(max(0, self.input_lags - 1), dtype=float)
        else:
            outputs, past_controls = history
            outputs = np.asarray(outputs, dtype=float).copy()
            outputs[0], outputs[1] = float(y_t), float(y_tm1)
        controls = np.concatenate(
            (
                np.asarray([float(u_t)], dtype=float),
                np.asarray(past_controls, dtype=float)[: self.input_lags - 1],
            )
        )
        if controls.size < self.input_lags:
            controls = np.pad(controls, (0, self.input_lags - controls.size))
        return np.concatenate((outputs[: self.output_lags], controls))

    def transition_mean(self, z: np.ndarray) -> float:
        return super().transition_mean(self._base_regressor(z))

    def nominal_transition_mean(self, z: np.ndarray) -> float:
        return super().nominal_transition_mean(self._base_regressor(z))

    def shift(self, z: np.ndarray, y_next: float, u_next: float) -> np.ndarray:
        if self.output_lags == 2 and self.input_lags == 1:
            return super().shift(z, y_next, u_next)
        z = np.asarray(z, dtype=float)
        outputs = np.concatenate(
            (np.asarray([float(y_next)]), z[: self.output_lags - 1])
        )
        current_and_past_controls = z[
            self.output_lags : self.output_lags + self.input_lags
        ]
        controls = np.concatenate(
            (
                np.asarray([float(u_next)]),
                current_and_past_controls[: self.input_lags - 1],
            )
        )
        return np.concatenate((outputs, controls))

    def step(self, z: np.ndarray, noise: bool = True) -> float:
        y_next = float(super().step(z, noise=noise))
        if self.output_lags > 2 or self.input_lags > 1:
            z = np.asarray(z, dtype=float)
            outputs = np.concatenate(
                (np.asarray([y_next]), z[: self.output_lags - 1])
            )
            controls = z[
                self.output_lags : self.output_lags + self.input_lags
            ]
            self._observed_histories[self._key(y_next, z[0])] = (
                outputs,
                controls[: self.input_lags - 1].copy(),
            )
        return y_next


experiment.NARXDoubleIntegratorEnv = LaggedNARXDoubleIntegratorEnv


if __name__ == "__main__":
    import main

    main.main()
