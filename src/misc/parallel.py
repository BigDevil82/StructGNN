import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskOutcome(Generic[T, R]):
    """统一的批处理任务结果，便于上层汇总成功/失败。"""

    item: T
    ok: bool
    result: Optional[R] = None
    error: Optional[str] = None


def run_batch(
    items: Sequence[T],
    worker: Callable[[T], R],
    max_workers: Optional[int] = None,
    backend: Literal["process", "thread"] = "process",
) -> List[TaskOutcome[T, R]]:
    """
    批量执行同构任务，支持多进程/多线程，并返回结构化结果。

    Notes:
        - Windows 下 `process` 模式要求 worker 为模块顶层函数（可被 pickle）。
        - 若任务量小于 2，将自动退化为串行执行，避免并行开销。
    """
    if not items:
        return []

    if len(items) < 2:
        outcomes: List[TaskOutcome[T, R]] = []
        for item in items:
            try:
                outcomes.append(TaskOutcome(item=item, ok=True, result=worker(item)))
            except Exception as e:
                outcomes.append(TaskOutcome(item=item, ok=False, error=str(e)))
        return outcomes

    if backend == "process" and _need_windows_process_sharding(max_workers):
        return _run_batch_with_windows_process_shards(items, worker, max_workers)

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    outcomes: List[TaskOutcome[T, R]] = []

    with executor_cls(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                outcomes.append(TaskOutcome(item=item, ok=True, result=future.result()))
            except Exception as e:
                outcomes.append(TaskOutcome(item=item, ok=False, error=str(e)))

    return outcomes


def _need_windows_process_sharding(max_workers: Optional[int]) -> bool:
    if os.name != "nt":
        return False
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    return requested > 61


def _run_batch_with_windows_process_shards(
    items: Sequence[T],
    worker: Callable[[T], R],
    max_workers: Optional[int],
) -> List[TaskOutcome[T, R]]:
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    shard_count = max(1, int(math.ceil(requested / 61)))

    buckets: list[list[T]] = [[] for _ in range(shard_count)]
    for idx, item in enumerate(items):
        buckets[idx % shard_count].append(item)

    shard_workers: list[int] = []
    base = requested // shard_count
    extra = requested % shard_count
    for i in range(shard_count):
        shard_workers.append(min(61, base + (1 if i < extra else 0)))

    outcomes: List[TaskOutcome[T, R]] = []
    with ThreadPoolExecutor(max_workers=shard_count) as launcher:
        futures = [
            launcher.submit(_run_batch_in_one_process_pool, bucket, worker, shard_workers[i])
            for i, bucket in enumerate(buckets)
            if bucket
        ]
        for future in as_completed(futures):
            outcomes.extend(future.result())

    return outcomes


def _run_batch_in_one_process_pool(
    items: Sequence[T],
    worker: Callable[[T], R],
    max_workers: int,
) -> List[TaskOutcome[T, R]]:
    outcomes: List[TaskOutcome[T, R]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                outcomes.append(TaskOutcome(item=item, ok=True, result=future.result()))
            except Exception as e:
                outcomes.append(TaskOutcome(item=item, ok=False, error=str(e)))
    return outcomes
