"""
Module implementing the Sample Entropy (SampEn) algorithm for online change-point detection.

This detector maintains a rolling window over a univariate time series, computes the
Sample Entropy of the current window, and emits a change-point when short-term
fluctuations in SampEn indicate a regime shift.

SampEn is computed as::

    SampEn(m, r, N) = -log( A / B )

where:
- ``m`` is the embedding (pattern) length,
- ``r`` is the tolerance radius (often a fraction of the windowed standard deviation),
- ``B`` is the number of matching pairs of m-length patterns under tolerance ``r``,
- ``A`` is the number of matching pairs of (m+1)-length patterns under tolerance ``r``.

This implementation supports a fixed ``r`` or an adaptive one via ``r_factor * std(window)``.
A detection is triggered when either:
1) the absolute change between consecutive SampEn values exceeds ``threshold``, or
2) the short-term variance of recent SampEn values, normalized by their mean, exceeds ``threshold``.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SampleEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Sample Entropy (SampEn).

    Parameters
    ----------
    window_size : int, default=100
        Sliding window length used to compute SampEn.
    m : int, default=2
        Embedding (pattern) length.
    r : float or None, default=None
        Fixed tolerance radius. If ``None``, an adaptive radius is used via ``r_factor``.
    r_factor : float, default=0.2
        Multiplicative factor for adaptive radius, i.e. ``r = r_factor * std(window)`` when ``r`` is ``None``.
    threshold : float, default=0.5
        Decision threshold used in both the first-order SampEn difference test and the
        normalized short-term variance test.

    Notes
    -----
    - The detector processes observations in a streaming fashion.
    - Change localization is returned as an approximate index around the center (or quarter)
      of the current window depending on which criterion was triggered.
    """

    def __init__(
        self,
        window_size: int = 100,
        m: int = 2,
        r: Optional[float] = None,
        r_factor: float = 0.2,
        threshold: float = 0.5,
    ):
        self._window_size = window_size
        self._m = m
        self._r = r
        self._r_factor = r_factor
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Ingest a new observation (or a batch) and update the internal detection state.

        Parameters
        ----------
        observation : float or ndarray of float
            A single value or a 1-D array of values to process sequentially.

        Returns
        -------
        bool
            ``True`` if a change-point was flagged after processing the input, ``False`` otherwise.
        """
        if isinstance(observation, np.ndarray):
            for obs in observation:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Ingest input and return the index of a detected change-point if present.

        Parameters
        ----------
        observation : float or ndarray of float
            A single value or a 1-D array of values to process.

        Returns
        -------
        int or None
            Estimated change-point index (0-based, relative to the processed stream),
            or ``None`` if no change-point is detected.
        """
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Process a single new observation and update the internal SampEn statistics.

        Parameters
        ----------
        observation : float
            New value to be appended to the rolling buffer.

        Returns
        -------
        None
        """
        v = 3
        self._buffer.append(observation)
        self._position += 1

        min_required = self._window_size + self._m
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_sample_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = float("inf")

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v - 1:
            prev_entropy = self._entropy_values[-2]
            curr_entropy = self._entropy_values[-1]

            # Robust difference that tolerates infinities from degenerate windows.
            if np.isinf(prev_entropy) and np.isinf(curr_entropy):
                entropy_diff = 0.0
            elif np.isinf(prev_entropy):
                entropy_diff = float("inf") if curr_entropy != 0 else 0.0
            elif np.isinf(curr_entropy):
                entropy_diff = float("inf") if prev_entropy != 0 else 0.0
            else:
                entropy_diff = abs(curr_entropy - prev_entropy)

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v + 2:
            recent_entropies = [e for e in self._entropy_values[-5:] if not np.isinf(e)]
            if len(recent_entropies) >= v:
                entropy_variance = np.var(recent_entropies)
                mean_entropy = np.mean(recent_entropies)
                if mean_entropy > 0 and entropy_variance / mean_entropy > self._threshold:
                    self._last_change_point = self._position - self._window_size // 4

    def _calculate_sample_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Sample Entropy for the given window.

        Parameters
        ----------
        time_series : ndarray of float, shape (N,)
            Current rolling window.

        Returns
        -------
        float
            The computed SampEn value. Returns ``inf`` when the input is too short or
            when there are no matches (degenerate case).

        Notes
        -----
        - If ``r`` is not provided, it is derived as ``r_factor * std(time_series)``.
        - When the window variance is zero, ``inf`` is returned to reflect undefined entropy.
        """
        N = len(time_series)
        if self._m + 1 > N:
            return float("inf")

        r = self._r
        if r is None:
            std_dev = float(np.std(time_series))
            if std_dev == 0:
                return float("inf")
            r = self._r_factor * std_dev

        assert r is not None

        B = self._count_matches(time_series, self._m, r)
        A = self._count_matches(time_series, self._m + 1, r)

        if B == 0 or A == 0:
            return float("inf")

        return float(-np.log(A / B))

    def _count_matches(self, time_series: npt.NDArray[np.float64], m: int, r: float) -> int:
        """
        Count the number of matching pairs of patterns under Chebyshev tolerance.

        Parameters
        ----------
        time_series : ndarray of float, shape (N,)
            Current rolling window.
        m : int
            Embedding length.
        r : float
            Tolerance radius.

        Returns
        -------
        int
            Number of unordered matching pairs among m-length sub-sequences.
        """
        N = len(time_series)
        matches = 0

        for i in range(N - m + 1):
            xi = time_series[i : i + m]
            for j in range(i + 1, N - m + 1):
                xj = time_series[j : j + m]
                distance = self._chebyshev_distance(xi, xj)
                if distance < r:
                    matches += 1

        return matches

    def _chebyshev_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Compute the Chebyshev (L-infinity) distance between two vectors.

        Parameters
        ----------
        x, y : ndarray of float, shape (m,)
            Two m-length vectors.

        Returns
        -------
        float
            ``max(|x - y|)``.
        """
        return float(np.max(np.abs(x - y)))

    def _euclidean_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Compute the Euclidean (L2) distance between two vectors.

        Parameters
        ----------
        x, y : ndarray of float, shape (m,)
            Two m-length vectors.

        Returns
        -------
        float
            ``sqrt(sum((x - y)^2))``.
        """
        return float(np.sqrt(np.sum((x - y) ** 2)))

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Sample Entropy values.

        Returns
        -------
        list of float
            A copy of the internal SampEn sequence evaluated at processed steps.
        """
        return self._entropy_values.copy()

    def get_current_r(self) -> Optional[float]:
        """
        Get the current tolerance radius ``r``.

        Returns
        -------
        float or None
            - If a fixed ``r`` was provided, it is returned.
            - If adaptive, returns ``r_factor * std(current_window)`` if available.
            - Otherwise, ``None``.
        """
        if self._r is not None:
            return self._r

        if len(self._buffer) > 0:
            current_window = np.array(list(self._buffer)[-self._window_size :])
            std_dev = np.std(current_window)
            return self._r_factor * std_dev if std_dev > 0 else None

        return None

    def reset(self) -> None:
        """
        Clear internal state and buffered statistics.

        Returns
        -------
        None
        """
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
