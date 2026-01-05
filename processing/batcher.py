from __future__ import annotations

"""Simple batching utilities to chunk iterables for model inference."""

from typing import Iterable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def batch_iterable(items: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yield lists of length <= batch_size from any iterable."""
    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batch_sequence(seq: Sequence[T], batch_size: int) -> Iterator[List[T]]:
    """Batch a sequence by slicing; slightly faster when len() is available."""
    n = len(seq)
    for i in range(0, n, batch_size):
        yield list(seq[i : i + batch_size])


__all__ = ["batch_iterable", "batch_sequence"]
