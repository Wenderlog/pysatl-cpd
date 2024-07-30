"""
MIT License

Copyright (c) 2024 Alexey Tatyanenko

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from abc import ABC, abstractmethod

import numpy.typing as npt


class ILikelihood(ABC):
    """
    Likelihood function's abstract base class.
    """

    @abstractmethod
    def learn(self, learning_sample: list[float]) -> None:
        """
        Learns first parameters of a likelihood function on a given sample.
        :param learning_sample: a sample for parameter learning.
        :return:
        """
        ...

    @abstractmethod
    def predict(self, observation: float) -> npt.ArrayLike:
        """
        Returns predictive probabilities for a given observation based on stored parameters.
        :param observation: an observation from a sample.
        :return: predictive probabilities for a given observation.
        """
        ...

    @abstractmethod
    def update(self, observation: float) -> None:
        """
        Updates parameters of a likelihood function according to the given observation.
        :param observation: an observation from a sample.
        :return:
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Clears likelihood function's state.
        :return:
        """
        ...
