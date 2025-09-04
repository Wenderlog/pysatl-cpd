"""
Module implementing the Slope Entropy algorithm for online change-point detection.

The detector slides a fixed-size window over a univariate time series, encodes
adjacent differences (slopes) into a finite alphabet using two slope thresholds
``delta`` and ``gamma``, estimates the entropy of the resulting slope-pattern
distribution, and triggers a change when short-term fluctuations in this entropy
exceed a decision threshold.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SlopeEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on slope symbolization and entropy.

    Parameters
    ----------
    window_size : int, default=100
        Sliding window length used to compute slope entropy.
    embedding_dim : int, default=3
        Length (in samples) of each subsequence used to form slope patterns.
        Each subsequence of length ``embedding_dim`` yields ``embedding_dim-1`` slope symbols.
    gamma : float, default=1.0
        Upper slope threshold. Slopes greater than ``gamma`` (or less than ``-gamma``) are encoded
        as the most extreme symbols ``2`` and ``-2`` respectively.
    delta : float, default=1e-3
        Inner slope threshold separating “flat/ties” from gentle slopes. Must satisfy ``0 <= delta < gamma``.
    threshold : float, default=0.3
        Decision threshold for triggering changes from short-term entropy dynamics.
    normalize : bool, default=True
        If ``True``, normalize entropy by the maximum possible entropy given the set of
        observed patterns in the current window (base-2).
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        gamma: float = 1.0,
        delta: float = 1e-3,
        threshold: float = 0.3,
        normalize: bool = True,
    ):
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._gamma = gamma
        self._delta = delta
        self._threshold = threshold
        self._normalize = normalize

        if delta >= gamma:
            raise ValueError(f"delta ({delta}) must be less than gamma ({gamma})")

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
        Process a single new observation and update slope-entropy statistics.

        Parameters
        ----------
        observation : float
            New value to be appended to the rolling buffer.

        Notes
        -----
        Detection logic combines three short-term tests:
        1) Absolute difference between the last two entropy values.
        2) Difference between means of two consecutive 5-sample entropy blocks.
        3) Difference between variances of two consecutive 4-sample entropy blocks.

        If any test exceeds its (scaled) threshold, a change-point is flagged and localized
        near the center or the end of the current window.
        """
        v = 2
        self._buffer.append(observation)
        self._position += 1

        min_required = self._window_size + self._embedding_dim - 1
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_slope_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v * 5:
            recent_window = self._entropy_values[-5:]
            previous_window = self._entropy_values[-10:-5]
            recent_mean = np.mean(recent_window)
            previous_mean = np.mean(previous_window)

            if abs(recent_mean - previous_mean) > self._threshold * 0.8:
                self._last_change_point = self._position - 2

        if len(self._entropy_values) >= v * 4:
            recent_variance = np.var(self._entropy_values[-4:])
            previous_variance = np.var(self._entropy_values[-8:-4])
            if abs(recent_variance - previous_variance) > self._threshold * 0.5:
                self._last_change_point = self._position - 1

    def _calculate_slope_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute slope entropy for the given window.

        Parameters
        ----------
        time_series : ndarray of float, shape (N,)
            Current rolling window.

        Returns
        -------
        float
            Base-2 entropy of the empirical distribution of slope patterns. Returns
            ``0.0`` if the window is too short or if no patterns can be formed.

        Notes
        -----
        - A slope pattern is formed by encoding the ``embedding_dim-1`` adjacent
          differences inside each length-``embedding_dim`` subsequence.
        - If ``normalize=True``, the entropy is divided by the maximum possible
          (``log2(#observed_patterns)``) to yield a value in ``[0, 1]``.
        """
        N = len(time_series)
        if self._embedding_dim > N:
            return 0.0

        pattern_counts: dict[tuple[int, ...], int] = {}
        total_patterns = 0

        for j in range(N - self._embedding_dim + 1):
            subsequence = time_series[j : j + self._embedding_dim]
            slope_pattern = self._create_slope_pattern(subsequence)
            pattern_key = tuple(slope_pattern)
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            total_patterns += 1

        if total_patterns == 0:
            return 0.0

        entropy = 0.0
        for count in pattern_counts.values():
            probability = count / total_patterns
            if probability > 0:
                entropy -= probability * np.log2(probability)

        if self._normalize:
            max_entropy = np.log2(len(pattern_counts)) if len(pattern_counts) > 1 else 1.0
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(entropy)

    def _create_slope_pattern(self, subsequence: npt.NDArray[np.float64]) -> list[int]:
        """
        Encode a length-``embedding_dim`` subsequence into a slope pattern.

        Parameters
        ----------
        subsequence : ndarray of float, shape (embedding_dim,)
            Subsequence from the current window.

        Returns
        -------
        list of int
            A list of length ``embedding_dim-1`` with symbols in ``{-2, -1, 0, 1, 2}``.

        Encoding
        --------
        For a slope ``d = x[i] - x[i-1]``:
        - ``d >  gamma`` → ``2``  (steep positive)
        - ``delta < d <= gamma`` → ``1``  (gentle positive)
        - ``|d| <= delta`` → ``0``  (flat/ties)
        - ``-gamma <= d < -delta`` → ``-1`` (gentle negative)
        - ``d < -gamma`` → ``-2`` (steep negative)
        """
        pattern = []
        for i in range(1, len(subsequence)):
            slope = subsequence[i] - subsequence[i - 1]

            if slope > self._gamma:
                symbol = 2
            elif self._delta < slope <= self._gamma:
                symbol = 1
            elif abs(slope) <= self._delta:
                symbol = 0
            elif -self._gamma <= slope < -self._delta:
                symbol = -1
            else:
                symbol = -2

            pattern.append(symbol)

        return pattern

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed slope-entropy values.

        Returns
        -------
        list of float
            A copy of the internal slope-entropy sequence evaluated at processed steps.
        """
        return self._entropy_values.copy()

    def get_current_parameters(self) -> dict[str, Any]:
        """
        Get the current configuration of the detector.

        Returns
        -------
        dict
            Dictionary with the current settings and derived limits.
        """
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "gamma": self._gamma,
            "delta": self._delta,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_symbols": 5,
            "max_patterns": 5 ** (self._embedding_dim - 1),
        }

    def set_parameters(
        self,
        embedding_dim: Optional[int] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        """
        Update detector parameters in-place.

        Parameters
        ----------
        embedding_dim : int, optional
            New subsequence length.
        gamma : float, optional
            New upper slope threshold.
        delta : float, optional
            New inner slope threshold (must remain strictly less than ``gamma``).
        threshold : float, optional
            New decision threshold for change detection.
        normalize : bool, optional
            Whether to normalize entropy to ``[0, 1]``.

        Raises
        ------
        ValueError
            If the updated ``delta`` is not strictly less than ``gamma``.
        """
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        if gamma is not None:
            self._gamma = gamma
        if delta is not None:
            self._delta = delta
        if threshold is not None:
            self._threshold = threshold
        if normalize is not None:
            self._normalize = normalize

        if self._delta >= self._gamma:
            raise ValueError(f"delta ({self._delta}) must be less than gamma ({self._gamma})")

    def get_pattern_distribution(self) -> dict[tuple[int, ...], float]:
        """
        Estimate the probability distribution over slope patterns in the current window.

        Returns
        -------
        dict
            Mapping from pattern (tuple of ints) to its empirical probability.
            Returns an empty dict if the buffer has fewer than ``window_size`` samples.
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        pattern_counts: dict[tuple[int, ...], int] = {}
        total_patterns = 0

        for j in range(len(current_window) - self._embedding_dim + 1):
            subsequence = current_window[j : j + self._embedding_dim]
            slope_pattern = self._create_slope_pattern(subsequence)
            pattern_key = tuple(slope_pattern)
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            total_patterns += 1

        pattern_probs = {}
        for pattern, count in pattern_counts.items():
            pattern_probs[pattern] = count / total_patterns if total_patterns > 0 else 0.0

        return pattern_probs

    def analyze_slope_characteristics(self) -> dict[str, Any]:
        """
        Compute descriptive statistics of slopes within the current window.

        Returns
        -------
        dict
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        slopes = np.diff(current_window)

        steep_positive = np.sum(slopes > self._gamma)
        gentle_positive = np.sum((slopes > self._delta) & (slopes <= self._gamma))
        flat = np.sum(np.abs(slopes) <= self._delta)
        gentle_negative = np.sum((slopes < -self._delta) & (slopes >= -self._gamma))
        steep_negative = np.sum(slopes < -self._gamma)
        total_slopes = len(slopes)

        return {
            "slope_entropy": self._calculate_slope_entropy(current_window),
            "steep_positive_ratio": steep_positive / total_slopes if total_slopes > 0 else 0,
            "gentle_positive_ratio": gentle_positive / total_slopes if total_slopes > 0 else 0,
            "flat_ratio": flat / total_slopes if total_slopes > 0 else 0,
            "gentle_negative_ratio": gentle_negative / total_slopes if total_slopes > 0 else 0,
            "steep_negative_ratio": steep_negative / total_slopes if total_slopes > 0 else 0,
            "slope_variance": np.var(slopes),
            "slope_mean": np.mean(slopes),
            "slope_std": np.std(slopes),
            "total_patterns": len(self.get_pattern_distribution()),
        }

    def get_symbol_meanings(self) -> dict[int, str]:
        """
        Return a human-readable legend for the slope symbols.

        Returns
        -------
        dict
            Mapping from symbol to description.
        """
        return {
            2: f"Steep positive slope (> {self._gamma})",
            1: f"Gentle positive slope ({self._delta} to {self._gamma})",
            0: f"Flat/ties (≤ {self._delta})",
            -1: f"Gentle negative slope (-{self._gamma} to -{self._delta})",
            -2: f"Steep negative slope (< -{self._gamma})",
        }

    def demonstrate_encoding(
        self, sample_data: list[float]
    ) -> dict[str, Union[str, float, int, list[float], list[int], list[list[int]], dict[int, str]]]:
        """
        Demonstrate slope symbolization and pattern construction for a small sample.

        Parameters
        ----------
        sample_data : list of float
            Example sequence to encode. Must have length ≥ ``embedding_dim``.

        Returns
        -------
        dict
            Keys:
            - ``original_data``: the input sequence,
            - ``slopes``: first differences,
            - ``symbols``: slope symbols for each difference,
            - ``patterns``: list of slope patterns of length ``embedding_dim-1``,
            - ``slope_entropy``: slope entropy over the sample,
            - ``encoding_rules``: symbol legend.
            If input is too short, returns ``{"error": "..."} ``.
        """
        if len(sample_data) < self._embedding_dim:
            return {"error": "Sample data too short"}

        sample_array = np.array(sample_data)
        slopes = np.diff(sample_array)

        symbols = []
        for slope in slopes:
            if slope > self._gamma:
                symbols.append(2)
            elif self._delta < slope <= self._gamma:
                symbols.append(1)
            elif abs(slope) <= self._delta:
                symbols.append(0)
            elif -self._gamma <= slope < -self._delta:
                symbols.append(-1)
            else:
                symbols.append(-2)

        patterns = []
        for j in range(len(sample_data) - self._embedding_dim + 1):
            subsequence = sample_array[j : j + self._embedding_dim]
            pattern = self._create_slope_pattern(subsequence)
            patterns.append(pattern)

        return {
            "original_data": sample_data,
            "slopes": slopes.tolist(),
            "symbols": symbols,
            "patterns": patterns,
            "slope_entropy": self._calculate_slope_entropy(sample_array),
            "encoding_rules": self.get_symbol_meanings(),
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
