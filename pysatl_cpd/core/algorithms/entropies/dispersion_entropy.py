"""
Module implementing the Dispersion Entropy (DisEn) algorithm for online change-point detection.

The detector maintains a rolling window over a univariate time series, maps values to
``c`` discrete classes via the Gaussian CDF, forms dispersion patterns of length ``m``
with time delay ``τ``, and computes the Shannon entropy of the pattern distribution::

    DisEn(m, c, τ) = - Σ p(pattern) * log(p(pattern))

where ``c`` is the number of classes, ``m`` is the embedding dimension, and ``τ`` is the delay.
Optionally, the entropy can be normalized by the maximal value ``log(c^m)``.

A change-point is signaled when any of the following holds:
1) The absolute difference between consecutive entropy values exceeds ``threshold``;
2) The short-term variance of entropy exceeds ``threshold``;
3) The mean shift between two consecutive short windows of entropy exceeds ``1.5 * threshold``.

The implementation supports streaming (online) processing and returns approximate
change-point indices relative to the processed stream.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class DispersionEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Dispersion Entropy (DisEn).

    Parameters
    ----------
    window_size : int, default=100
        Sliding window length used for entropy computation.
    embedding_dim : int, default=3
        Embedding dimension ``m`` of dispersion patterns.
    num_classes : int, default=6
        Number of discrete classes ``c`` used to quantize the window via Gaussian CDF.
    time_delay : int, default=1
        Delay ``τ`` between consecutive elements of a dispersion pattern.
    threshold : float, default=0.2
        Decision threshold used in the detection criteria.
    normalize : bool, default=True
        If True, normalize entropy by ``log(c^m)``.

    Notes
    -----
    - If ``c^m >= window_size``, the pattern space becomes too large for reliable
      estimation from the window; a ``ValueError`` is raised.
    - Observations are processed online; change localization is approximate and tied
      to the center/quarter of the active window depending on which criterion triggers.
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        num_classes: int = 6,
        time_delay: int = 1,
        threshold: float = 0.2,
        normalize: bool = True,
    ):
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._num_classes = num_classes
        self._time_delay = time_delay
        self._threshold = threshold
        self._normalize = normalize

        if num_classes**embedding_dim >= window_size:
            raise ValueError(
                f"c^w ({num_classes}^{embedding_dim} = {num_classes**embedding_dim}) "
                f"should be less than window_size ({window_size})"
            )

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
        Process a single new observation and update the internal DisEn statistics.

        Parameters
        ----------
        observation : float
            New value to be appended to the rolling buffer.

        Returns
        -------
        None
        """
        self._buffer.append(observation)
        self._position += 1
        v = 10  # window length (in entropy samples) for mean-shift check
        v1 = 2  # minimal length (in entropy samples) for first-difference check

        min_required = self._window_size + (self._embedding_dim - 1) * self._time_delay
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_dispersion_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        # Criterion 1: first-order entropy difference
        if len(self._entropy_values) >= v1:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        # Criterion 2: short-term variance of entropy
        if len(self._entropy_values) >= v1 + 3:
            recent_entropies = self._entropy_values[-5:]
            entropy_variance = np.var(recent_entropies)
            # mean not used in the current rule; kept for parity with ApEn/SampEn variants
            _ = np.mean(recent_entropies)
            if entropy_variance > self._threshold:
                self._last_change_point = self._position - self._window_size // 4

        # Criterion 3: mean shift between two adjacent windows of entropy
        if len(self._entropy_values) >= v:
            window1 = self._entropy_values[-10:-5]
            window2 = self._entropy_values[-5:]
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)
            if abs(mean2 - mean1) > self._threshold * 1.5:
                self._last_change_point = self._position - 2

    def _calculate_dispersion_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Dispersion Entropy for the given window.

        Steps
        -----
        1) Standardize to ``N(μ, o)`` and apply Gaussian CDF to obtain ``y in (0, 1)``;
        2) Discretize into ``c`` classes to get integer-coded series ``z`` in ``{1, ..., c}``;
        3) Form dispersion patterns of length ``m`` with delay ``τ``;
        4) Compute the Shannon entropy of pattern probabilities and (optionally) normalize.

        Parameters
        ----------
        time_series : ndarray of float, shape (N,)
            Current rolling window.

        Returns
        -------
        float
            Dispersion Entropy value for the window. Returns ``0.0`` when input is too short or degenerate.
        """
        N = len(time_series)
        if self._embedding_dim > N:
            return 0.0

        mu = np.mean(time_series)
        sigma = np.std(time_series)
        if sigma == 0:
            return 0.0

        y_series = norm.cdf(time_series, loc=mu, scale=sigma)
        z_series = self._discretize_to_classes(y_series)
        patterns = self._create_dispersion_patterns(z_series)
        pattern_probs = self._calculate_pattern_probabilities(patterns)
        de_value = self._calculate_shannon_entropy(pattern_probs)

        if self._normalize:
            max_entropy = np.log(self._num_classes**self._embedding_dim)
            if max_entropy > 0:
                de_value = de_value / max_entropy

        return float(de_value)

    def _discretize_to_classes(self, y_series: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Map continuous values ``y in (0, 1)`` to integer classes ``{1, ..., c}``.

        Parameters
        ----------
        y_series : ndarray of float, shape (N,)
            Values after applying the Gaussian CDF to the window.

        Returns
        -------
        ndarray of int32, shape (N,)
            Discrete class labels in the range ``[1, c]``.
        """
        rc_values = self._num_classes * y_series + 0.5
        z_series = np.round(rc_values).astype(np.int32)
        z_series = np.clip(z_series, 1, self._num_classes)
        return z_series

    def _create_dispersion_patterns(self, z_series: npt.NDArray[np.int32]) -> list[tuple[int, ...]]:
        """
        Construct dispersion patterns of length ``m`` with delay ``τ`` from class labels.

        Parameters
        ----------
        z_series : ndarray of int32, shape (N,)
            Discrete class labels in ``[1, c]``.

        Returns
        -------
        list of tuple[int, ...]
            List of length-``m`` patterns encoded as integer tuples.
        """
        N = len(z_series)
        patterns = []

        for i in range(N - (self._embedding_dim - 1) * self._time_delay):
            pattern = []
            for j in range(self._embedding_dim):
                idx = i + j * self._time_delay
                if idx < N:
                    pattern.append(z_series[idx])
            if len(pattern) == self._embedding_dim:
                patterns.append(tuple(pattern))

        return patterns

    def _calculate_pattern_probabilities(self, patterns: list[tuple[int, ...]]) -> dict[tuple[int, ...], float]:
        """
        Estimate pattern probabilities from the list of patterns.

        Parameters
        ----------
        patterns : list of tuple[int, ...]
            Dispersion patterns extracted from the current window.

        Returns
        -------
        dict[tuple[int, ...], float]
            Mapping from pattern to its empirical probability.
        """
        if not patterns:
            return {}

        pattern_counts: dict[tuple[int, ...], int] = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        total_patterns = len(patterns)
        pattern_probs: dict[tuple[int, ...], float] = {}
        for pattern, count in pattern_counts.items():
            pattern_probs[pattern] = count / total_patterns

        return pattern_probs

    def _calculate_shannon_entropy(self, pattern_probs: dict[tuple[int, ...], float]) -> float:
        """
        Compute Shannon entropy from pattern probabilities.

        Parameters
        ----------
        pattern_probs : dict[tuple[int, ...], float]
            Empirical probability distribution of dispersion patterns.

        Returns
        -------
        float
            ``- Σ p * log(p)`` (natural logarithm).
        """
        if not pattern_probs:
            return 0.0

        entropy = 0.0
        for prob in pattern_probs.values():
            if prob > 0:
                entropy -= prob * np.log(prob)

        return entropy

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Dispersion Entropy values.

        Returns
        -------
        list of float
            A copy of the internal DisEn sequence evaluated at processed steps.
        """
        return self._entropy_values.copy()

    def get_current_parameters(self) -> dict[str, int | float | bool]:
        """
        Return a snapshot of the current algorithm parameters and derived capacities.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``window_size``
            - ``embedding_dim``
            - ``num_classes``
            - ``time_delay``
            - ``threshold``
            - ``normalize``
            - ``max_patterns`` = ``c^m`` (pattern capacity)
        """
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "num_classes": self._num_classes,
            "time_delay": self._time_delay,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_patterns": self._num_classes**self._embedding_dim,
        }

    def set_parameters(
        self,
        embedding_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        time_delay: Optional[int] = None,
        threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        """
        Update detector hyperparameters in place.

        Parameters
        ----------
        embedding_dim : int or None, optional
            New embedding dimension ``m``.
        num_classes : int or None, optional
            New number of classes ``c``.
        time_delay : int or None, optional
            New delay ``τ`` between pattern elements.
        threshold : float or None, optional
            New decision threshold.
        normalize : bool or None, optional
            Whether to normalize entropy values.

        Raises
        ------
        ValueError
            If the updated configuration violates ``c^m < window_size``.
        """
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        if num_classes is not None:
            self._num_classes = num_classes
        if time_delay is not None:
            self._time_delay = time_delay
        if threshold is not None:
            self._threshold = threshold
        if normalize is not None:
            self._normalize = normalize

        if self._num_classes**self._embedding_dim >= self._window_size:
            raise ValueError(
                f"c^w ({self._num_classes}^{self._embedding_dim}) should be less than window_size ({self._window_size})"
            )

    def get_pattern_distribution(self) -> dict[tuple[int, ...], int]:
        """
        Return pattern counts for the current window (for diagnostics/inspection).

        Returns
        -------
        dict[tuple[int, ...], int]
            Mapping from pattern to its count in the current window.
            Returns an empty dict if the buffer is not yet filled enough.
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        mu = np.mean(current_window)
        sigma = np.std(current_window)
        if sigma == 0:
            return {}

        y_series = norm.cdf(current_window, loc=mu, scale=sigma)
        z_series = self._discretize_to_classes(y_series)
        patterns = self._create_dispersion_patterns(z_series)

        pattern_counts: dict[tuple[int, ...], int] = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return pattern_counts

    def analyze_complexity(self) -> dict[str, Union[float, int]]:
        """
        Compute diagnostic complexity measures for the current window.

        Returns
        -------
        dict
            Diagnostic metrics including:
            - ``dispersion_entropy``: current DisEn value
            - ``normalized_entropy``: DisEn / log(c^m)
            - ``unique_patterns``: number of distinct patterns observed
            - ``max_possible_patterns``: ``c^m``
            - ``pattern_diversity``: unique / max_possible
            - ``window_std``: standard deviation of the window
            - ``window_mean``: mean of the window

            Returns an empty dict if the buffer is not yet filled enough.
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_dispersion_entropy(current_window)

        pattern_dist = self.get_pattern_distribution()
        unique_patterns = len(pattern_dist)
        max_possible_patterns = self._num_classes**self._embedding_dim

        pattern_diversity = unique_patterns / max_possible_patterns if max_possible_patterns > 0 else 0

        max_entropy = np.log(max_possible_patterns) if max_possible_patterns > 0 else 1
        relative_entropy = current_entropy / max_entropy if max_entropy > 0 else 0

        return {
            "dispersion_entropy": current_entropy,
            "normalized_entropy": relative_entropy,
            "unique_patterns": unique_patterns,
            "max_possible_patterns": max_possible_patterns,
            "pattern_diversity": pattern_diversity,
            "window_std": np.std(current_window),
            "window_mean": np.mean(current_window),
        }

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
