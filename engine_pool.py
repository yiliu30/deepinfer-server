"""Multi-rank DeepInfer engine pool used by the Claude Code gateway.

Each rank runs in its own process with one visible GPU. All ranks share one
expert-parallel world, and the driver thread calls ``step`` on every rank in
lockstep so expert-parallel collectives always agree. Ranks without requests
run DeepInfer's placeholder step, which keeps them inside the collectives.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
import uuid

READY_TIMEOUT_S = 1200.0
STEP_TIMEOUT_S = 1800.0


def _rank_main(rank, device, cfg, in_q, out_q, ready_evt):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        sys.path.insert(0, os.path.join(cfg["repo"], "examples"))
        import _config

        _config.setup_common_env(
            model_name=cfg["model"],
            sampler_force_topk=cfg["sampler_force_topk"],
            set_override_experts=True,
            enable_eplb=True,
        )

        import torch  # noqa: F401  (must precede deepinfer's C extensions)
        import numpy as np
        from deepinfer._C.engine import RequestOutput
        from deepinfer.entrypoints.llm import LLM
        from deepinfer.sampling_params import SamplingParams

        llm = LLM(
            model=cfg["model"],
            tokenizer=cfg["model"],
            trust_remote_code=True,
            expert_parallel_size=cfg["ep"],
            context_parallel_size=1,
            engine_rank=rank,
            data_parallel_size=cfg["dp"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            decode_sequence=False,
            chunked_prefill=True,
            disable_log_stats=True,
            max_model_len=cfg["max_model_len"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            max_num_seqs=cfg["max_num_seqs"],
            lookahead_tokens=cfg["lookahead_tokens"],
        )
        engine = llm.llm_engine

        def dump(outputs):
            items = []
            for raw in outputs:
                out = RequestOutput.clone(raw)
                items.append(
                    {
                        "request_id": out.request_id,
                        "token_id": int(out.token_id),
                        "finished": bool(out.finished),
                        "finish_reason": out.finish_reason,
                    }
                )
            return items

        out_q.put(("ready", rank, None))
        ready_evt.set()

        while True:
            cmd = in_q.get()
            kind = cmd[0]
            if kind == "add":
                _, request_id, token_ids, sp = cmd
                engine.add_request(
                    user_id="0",
                    request_id=request_id,
                    sampling_params=SamplingParams(**sp),
                    prompt_token_ids=np.asarray(token_ids, dtype=np.int32),
                    multi_modal_data={"image_bytes": [], "image_sizes": []},
                )
                out_q.put(("added", rank, request_id))
            elif kind == "step":
                out_q.put(("step", rank, dump(engine.step())))
            elif kind == "flush":
                out_q.put(("flush", rank, dump(engine.flush_inflight_reqs())))
            elif kind == "abort":
                try:
                    engine.abort_request(cmd[1])
                except Exception:
                    traceback.print_exc()
                out_q.put(("aborted", rank, cmd[1]))
            elif kind == "close":
                engine.close()
                out_q.put(("closed", rank, None))
                return
            else:
                out_q.put(("error", rank, f"unknown command {kind!r}"))
    except Exception:
        out_q.put(("error", rank, traceback.format_exc()))
        ready_evt.set()
        raise


class EnginePool:
    """Drive four DeepInfer ranks and route token streams to HTTP requests."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_ranks = len(cfg["devices"])
        self._ctx = mp.get_context("spawn")
        self._in_qs = []
        self._out_qs = []
        self._ready_evts = []
        self._procs = []
        self._routes = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._need_flush = False
        self._driver = None
        self.error = None
        self.started = False

    def start(self):
        for rank, device in enumerate(self.cfg["devices"]):
            in_q = self._ctx.Queue()
            out_q = self._ctx.Queue()
            ready_evt = self._ctx.Event()
            proc = self._ctx.Process(
                target=_rank_main,
                args=(rank, device, self.cfg, in_q, out_q, ready_evt),
                daemon=False,
            )
            proc.start()
            self._in_qs.append(in_q)
            self._out_qs.append(out_q)
            self._ready_evts.append(ready_evt)
            self._procs.append(proc)

        deadline = time.time() + READY_TIMEOUT_S
        for rank, out_q in enumerate(self._out_qs):
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.error = f"rank {rank} did not become ready in time"
                    raise TimeoutError(self.error)
                try:
                    kind, got_rank, payload = out_q.get(timeout=min(5.0, remaining))
                except queue.Empty:
                    if not self._procs[rank].is_alive():
                        self.error = f"rank {rank} exited during startup"
                        raise RuntimeError(self.error)
                    continue
                if kind == "ready":
                    break
                if kind == "error":
                    self.error = f"rank {rank} startup failed:\n{payload}"
                    raise RuntimeError(self.error)

        self._driver = threading.Thread(target=self._driver_loop, name="engine-driver", daemon=True)
        self._driver.start()
        self.started = True

    def _drain_acks(self, rank, out_q, wanted):
        while True:
            kind, got_rank, payload = out_q.get(timeout=STEP_TIMEOUT_S)
            if kind == "error":
                raise RuntimeError(f"rank {got_rank} failed:\n{payload}")
            if kind == wanted:
                return payload

    def _run_round(self, kind):
        for in_q in self._in_qs:
            in_q.put((kind,))
        results = []
        for rank, out_q in enumerate(self._out_qs):
            results.append(self._drain_acks(rank, out_q, kind))
        return results

    def _driver_loop(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    active = bool(self._routes)
                    need_flush = self._need_flush
                if not active:
                    if need_flush:
                        self._run_round("flush")
                        with self._lock:
                            self._need_flush = False
                    else:
                        time.sleep(0.002)
                    continue
                for items in self._run_round("step"):
                    for item in items:
                        with self._lock:
                            route = self._routes.get(item["request_id"])
                        if route is not None:
                            route[1].put(item)
        except Exception:
            self.error = traceback.format_exc()
            self._stop.set()
            with self._lock:
                routes = list(self._routes.values())
            for _, q in routes:
                q.put(None)

    def submit(self, token_ids, sampling_params):
        with self._lock:
            counts = [0] * self.n_ranks
            for rank, _ in self._routes.values():
                counts[rank] += 1
            rank = counts.index(min(counts))
            request_id = uuid.uuid4().hex
            q = queue.Queue(maxsize=8192)
            self._routes[request_id] = (rank, q)
        self._in_qs[rank].put(("add", request_id, list(token_ids), sampling_params))
        return request_id, q

    def release(self, request_id, abort):
        with self._lock:
            route = self._routes.pop(request_id, None)
            if not self._routes:
                self._need_flush = True
        if route is not None and abort:
            self._in_qs[route[0]].put(("abort", request_id))

    def close(self):
        self._stop.set()
        if self._driver is not None:
            self._driver.join(timeout=10.0)
        for in_q in self._in_qs:
            try:
                in_q.put(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=30.0)
