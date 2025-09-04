"""
Module implementing the Approximate Entropy (ApEn) algorithm for online change-point detection.

The algorithm maintains a rolling window over a univariate time series, computes the
Approximate Entropy for the current window, and raises a change-point signal when
the short-term dynamics of ApEn indicate a regime shift.

ApEn is computed as::

    ApEn(m, r, N) = phi(m, r) - phi(m+1, r)

where ``m`` is the embedding (pattern) length and ``r`` is a tolerance radius
(often expressed as a fraction of the windowed standard deviation).

This implementation supports a fixed ``r`` or an adaptive one via ``r_factor * std(window)``.
A detection is triggered when either:
1) the absolute change in consecutive ApEn values exceeds ``threshold``, or
2) the short-term variance of ApEn, normalized by its mean magnitude, exceeds ``threshold``.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ApproximateEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Approximate Entropy (ApEn).

    Parameters
    ----------
    window_size : int, default=100
        Sliding window length used to compute ApEn.
    m : int, default=2
        Embedding (pattern) length for ApEn.
    r : float or None, default=None
        Fixed tolerance radius. If ``None``, an adaptive radius is used via ``r_factor``.
    r_factor : float, default=0.2
        Multiplicative factor for adaptive radius, i.e. ``r = r_factor * std(window)`` when ``r`` is ``None``.
    threshold : float, default=0.3
        Decision threshold used in both the first-order ApEn difference test and the
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
        threshold: float = 0.3,
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
        Process a single new observation and update the internal ApEn statistics.

        Parameters
        ----------
        observation : float
            New value to be appended to the rolling buffer.

        Returns
        -------
        None
        """
        v = 2
        self._buffer.append(observation)
        self._position += 1

        min_required = self._window_size + self._m
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_approximate_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v + 3:
            recent_entropies = self._entropy_values[-5:]
            entropy_variance = np.var(recent_entropies)
            mean_entropy = np.mean(recent_entropies)
            if mean_entropy > 0 and entropy_variance / abs(mean_entropy) > self._threshold:
                self._last_change_point = self._position - self._window_size // 4

    def _calculate_approximate_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Approximate Entropy for the given window.

        Parameters
        ----------
        time_series : ndarray of float, shape (N,)
            Current rolling window.

        Returns
        -------
        float
            The computed ApEn value. Returns ``0.0`` when the input is too short or degenerate.

        Notes
        -----
        - If ``r`` is not provided, it is derived as ``r_factor * std(time_series)``.
        - When the window variance is zero, ``0.0`` is returned to avoid division by zero.
        """
        N = len(time_series)

        if self._m + 1 > N:
            return 0.0

        r = self._r
        if r is None:
            std_dev = float(np.std(time_series))
            if std_dev == 0:
                return 0.0
            r = self._r_factor * std_dev

        assert r is not None

        phi_m = self._calculate_phi(time_series, self._m, r)
        phi_m_plus_1 = self._calculate_phi(time_series, self._m + 1, r)
        approximate_entropy = phi_m - phi_m_plus_1
        return float(approximate_entropy)

    def _calculate_phi(self, time_series: npt.NDArray[np.float64], m: int, r: float) -> float:
        """
        Compute the auxiliary ``phi(m, r)`` term for ApEn.

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
        float
            The average log of match proportions across all m-patterns in the window.

        Notes
        -----
        - Uses Chebyshev (max-abs) distance between m-length vectors.
        - Protects ``log(0)`` by adding a small epsilon.
        """
        N = len(time_series)
        n = N - m + 1

        if n <= 0:
            return 0.0

        log_sum = 0.0

        for i in range(n):
            xi = time_series[i : i + m]

            matches = 0
            for j in range(n):
                xj = time_series[j : j + m]
                distance = self._max_distance(xi, xj)

                if distance <= r:
                    matches += 1

            C_i_m = matches / n

            if C_i_m > 0:
                log_sum += np.log(C_i_m)
            else:
                log_sum += np.log(1e-10)

        phi = log_sum / n
        return float(phi)

    def _max_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
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
        Get the history of computed Approximate Entropy values.

        Returns
        -------
        list of float
            A copy of the internal ApEn sequence evaluated at processed steps.
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

    def get_pattern_length(self) -> int:
        """
        Get the current embedding length ``m``.

        Returns
        -------
        int
            The pattern length parameter used for ApEn.
        """
        return self._m

    def set_parameters(
        self,
        m: Optional[int] = None,
        r: Optional[float] = None,
        r_factor: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Update detector hyperparameters in place.

        Parameters
        ----------
        m : int or None, optional
            New embedding length. If ``None``, unchanged.
        r : float or None, optional
            New fixed tolerance. If ``None``, unchanged.
        r_factor : float or None, optional
            New adaptive multiplier. If ``None``, unchanged.
        threshold : float or None, optional
            New decision threshold. If ``None``, unchanged.

        Returns
        -------
        None
        """
        if m is not None:
            self._m = m
        if r is not None:
            self._r = r
        if r_factor is not None:
            self._r_factor = r_factor
        if threshold is not None:
            self._threshold = threshold

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
