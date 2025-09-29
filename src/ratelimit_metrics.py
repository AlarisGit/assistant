import time
import threading
import logging
import atexit
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.last = _now()
        self._lock = threading.Lock()

    def _refill(self, now: float):
        if self.tokens < self.capacity:
            dt = max(0.0, now - self.last)
            self.tokens = min(self.capacity, self.tokens + dt * self.refill_per_sec)
        self.last = now

    def try_consume(self, amount: float) -> bool:
        with self._lock:
            now = _now()
            self._refill(now)
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def time_to_avail(self, amount: float) -> float:
        with self._lock:
            now = _now()
            self._refill(now)
            if self.tokens >= amount:
                return 0.0
            needed = amount - self.tokens
            if self.refill_per_sec <= 0:
                return float('inf')
            return needed / self.refill_per_sec


class _RateLimitState:
    def __init__(self):
        self._lock = threading.Lock()
        # buckets[(provider, model)] = { 'rpm': TokenBucket|None, 'tpm': TokenBucket|None }
        self.buckets: Dict[Tuple[str, str], Dict[str, Optional[TokenBucket]]] = {}

    def _get_limits(self, provider: str, model: str) -> Tuple[Optional[int], Optional[int]]:
        key = f"{model}@{provider}"
        cfg = getattr(config, 'RATE_LIMITS', {}) or {}
        val = cfg.get(key)
        if not val:
            return None, None
        rpm = val.get('rpm') if isinstance(val, dict) else None
        tpm = val.get('tpm') if isinstance(val, dict) else None
        return rpm, tpm

    def get_buckets(self, provider: str, model: str) -> Dict[str, Optional[TokenBucket]]:
        k = (provider, model)
        with self._lock:
            if k in self.buckets:
                return self.buckets[k]
            rpm, tpm = self._get_limits(provider, model)
            entry: Dict[str, Optional[TokenBucket]] = {'rpm': None, 'tpm': None}
            if rpm and rpm > 0:
                entry['rpm'] = TokenBucket(capacity=float(rpm), refill_per_sec=float(rpm) / 60.0)
            if tpm and tpm > 0:
                entry['tpm'] = TokenBucket(capacity=float(tpm), refill_per_sec=float(tpm) / 60.0)
            self.buckets[k] = entry
            return entry


_rate_state = _RateLimitState()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Heuristic: ~4 chars per token
    return max(1, int(len(text) / 4))


class MetricsAggregator:
    def __init__(self):
        self._lock = threading.Lock()
        # stats[(provider, model, action)] = {...}
        self.interval: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self.totals: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    def record(self, provider: str, model: str, action: str, latency_ms: float,
               in_tok: int, out_tok: int, error: bool, retries: int):
        key = (provider, model, action)
        with self._lock:
            for bucket in (self.interval, self.totals):
                s = bucket.get(key)
                if not s:
                    s = {
                        'req': 0, 'in_tok': 0, 'out_tok': 0,
                        'lat_sum': 0.0, 'lat_cnt': 0,
                        'errors': 0, 'retries': 0,
                    }
                    bucket[key] = s
                s['req'] += 1
                s['in_tok'] += int(in_tok)
                s['out_tok'] += int(out_tok)
                s['lat_sum'] += float(latency_ms)
                s['lat_cnt'] += 1
                if error:
                    s['errors'] += 1
                s['retries'] += int(retries)

    def snapshot_and_reset_interval(self):
        with self._lock:
            snap = self.interval
            self.interval = {}
            return snap

    def snapshot_totals(self):
        with self._lock:
            return dict(self.totals)




_metrics = MetricsAggregator()


class _Reporter:
    def __init__(self, interval_sec: int):
        self.interval = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self.interval <= 0:
            return
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="llm-metrics-reporter", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.wait(self.interval):
            snap = _metrics.snapshot_and_reset_interval()
            if not snap:
                continue
            
            # Get interval stats
            lines = ["provider     model        action   req  in_tok out_tok avg_lat ms out_tok/s",
                     "--------     -----        ------   ---  ------ ------- ---------- ---------"]
            for (prov, model, action), s in sorted(snap.items()):
                avg_lat = (s['lat_sum'] / max(1, s['lat_cnt']))
                # throughput based on avg latency per request (very rough)
                sec = max(0.001, avg_lat / 1000.0)
                tps = int(s['out_tok'] / max(1.0, (s['req'] * sec)))
                lines.append(f"{prov:<12} {model:<12} {action:<8} {int(s['req']):>3} {int(s['in_tok']):>6} {int(s['out_tok']):>7} {avg_lat:>10.0f} {tps:>9}")
            interval_table = "\n".join(lines)
            
            # Get cumulative totals
            totals_snap = _metrics.snapshot_totals()
            totals_lines = ["provider     model        action   req  in_tok out_tok avg_lat ms errors retries",
                           "--------     -----        ------   ---  ------ ------- ---------- ------ -------"]
            for (prov, model, action), s in sorted(totals_snap.items()):
                avg_lat = (s['lat_sum'] / max(1, s['lat_cnt']))
                totals_lines.append(f"{prov:<12} {model:<12} {action:<8} {int(s['req']):>3} {int(s['in_tok']):>6} {int(s['out_tok']):>7} {avg_lat:>10.0f} {int(s['errors']):>6} {int(s['retries']):>7}")
            totals_table = "\n".join(totals_lines)
            
            # Log both interval and cumulative stats
            logger.info("Operation stats (last interval)\n" + interval_table)
            logger.info("Progress totals (cumulative)\n" + totals_table)


_reporter = _Reporter(getattr(config, 'REPORT_INTERVAL_SECONDS', 60))
_reporter.start()


def _print_totals():
    snap = _metrics.snapshot_totals()
    if not snap:
        return
    lines = ["provider     model        action   req  in_tok out_tok avg_lat ms errors retries",
             "--------     -----        ------   ---  ------ ------- ---------- ------ -------"]
    for (prov, model, action), s in sorted(snap.items()):
        avg_lat = (s['lat_sum'] / max(1, s['lat_cnt']))
        lines.append(f"{prov:<12} {model:<12} {action:<8} {int(s['req']):>3} {int(s['in_tok']):>6} {int(s['out_tok']):>7} {avg_lat:>10.0f} {int(s['errors']):>6} {int(s['retries']):>7}")
    table = "\n".join(lines)
    logger.info("Totals (cumulative)\n" + table)


atexit.register(_print_totals)


class RateLimitedMetricsProvider:
    def __init__(self, inner: Any, provider_name: str):
        self.inner = inner
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.RateLimited[{provider_name}]")
        # Propagate api_key for compatibility with wrappers expecting it
        self.api_key = getattr(inner, 'api_key', '')

    def _enforce_limits(self, model: str, est_in_tokens: int):
        buckets = _rate_state.get_buckets(self.provider_name, model)
        # No buckets configured => unlimited
        if not buckets['rpm'] and not buckets['tpm']:
            return
        waits: List[float] = []
        if buckets['rpm']:
            if not buckets['rpm'].try_consume(1.0):
                waits.append(buckets['rpm'].time_to_avail(1.0))
        if buckets['tpm']:
            # Reserve input tokens; output will be adjusted post-call if needed (we don't put back)
            needed = float(max(0, est_in_tokens))
            if not buckets['tpm'].try_consume(needed):
                waits.append(buckets['tpm'].time_to_avail(needed))
        if waits:
            sleep_s = max(0.0, min(waits))
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _estimate_output_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def generate_text(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [],
                      image: str = '', action: Optional[str] = None, **kwargs) -> str:
        model = kwargs.get('model', '')
        # Estimate input tokens
        in_tok = _estimate_tokens(system_prompt) + _estimate_tokens(prompt)
        if history:
            for u, a in history:
                in_tok += _estimate_tokens(u) + _estimate_tokens(a)
        # Images are ignored for TPM by default
        self._enforce_limits(model, in_tok)

        t0 = _now()
        retries = 0
        try:
            resp = self.inner.generate_text(prompt, system_prompt, history, image, **kwargs)
            latency_ms = (_now() - t0) * 1000.0
            out_tok = self._estimate_output_tokens(resp)
            _metrics.record(self.provider_name, model, action or 'rsp', latency_ms, in_tok, out_tok, False, retries)
            return resp
        except Exception as e:
            # Basic 429 handling by message sniffing
            retry_after = 0.0
            msg = str(e).lower()
            if '429' in msg or 'rate limit' in msg:
                retry_after = 2.0
            if retry_after > 0 and retries < 2:
                retries += 1
                time.sleep(retry_after)
                try:
                    resp = self.inner.generate_text(prompt, system_prompt, history, image, **kwargs)
                    latency_ms = (_now() - t0) * 1000.0
                    out_tok = self._estimate_output_tokens(resp)
                    _metrics.record(self.provider_name, model, action or 'rsp', latency_ms, in_tok, out_tok, False, retries)
                    return resp
                except Exception as e2:
                    _metrics.record(self.provider_name, model, action or 'rsp', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
                    raise e2
            _metrics.record(self.provider_name, model, action or 'rsp', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
            raise

    def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        in_tok = _estimate_tokens(text)
        self._enforce_limits(model, in_tok)
        t0 = _now()
        retries = 0
        try:
            emb = self.inner.generate_embedding(text, model, **kwargs)
            latency_ms = (_now() - t0) * 1000.0
            _metrics.record(self.provider_name, model, 'emb', latency_ms, in_tok, 0, False, retries)
            return emb
        except Exception as e:
            msg = str(e).lower()
            retry_after = 0.0
            if '429' in msg or 'rate limit' in msg:
                retry_after = 2.0
            if retry_after > 0 and retries < 2:
                retries += 1
                time.sleep(retry_after)
                try:
                    emb = self.inner.generate_embedding(text, model, **kwargs)
                    latency_ms = (_now() - t0) * 1000.0
                    _metrics.record(self.provider_name, model, 'emb', latency_ms, in_tok, 0, False, retries)
                    return emb
                except Exception as e2:
                    _metrics.record(self.provider_name, model, 'emb', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
                    raise e2
            _metrics.record(self.provider_name, model, 'emb', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
            raise
    
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [],
                                 image: str = '', action: Optional[str] = None, **kwargs) -> str:
        """Generate text asynchronously with rate limiting and metrics"""
        model = kwargs.get('model', '')
        # Estimate input tokens
        in_tok = _estimate_tokens(system_prompt) + _estimate_tokens(prompt)
        if history:
            for u, a in history:
                in_tok += _estimate_tokens(u) + _estimate_tokens(a)
        # Images are ignored for TPM by default
        
        # Run rate limiting in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._enforce_limits(model, in_tok))

        t0 = _now()
        retries = 0
        try:
            resp = await self.inner.generate_text_async(prompt, system_prompt, history, image, **kwargs)
            latency_ms = (_now() - t0) * 1000.0
            out_tok = self._estimate_output_tokens(resp)
            _metrics.record(self.provider_name, model, action or 'rsp', latency_ms, in_tok, out_tok, False, retries)
            return resp
        except Exception as e:
            # Basic 429 handling by message sniffing
            retry_after = 0.0
            msg = str(e).lower()
            if '429' in msg or 'rate limit' in msg:
                retry_after = 2.0
            if retry_after > 0 and retries < 2:
                retries += 1
                await asyncio.sleep(retry_after)  # Use async sleep
                try:
                    resp = await self.inner.generate_text_async(prompt, system_prompt, history, image, **kwargs)
                    latency_ms = (_now() - t0) * 1000.0
                    out_tok = self._estimate_output_tokens(resp)
                    _metrics.record(self.provider_name, model, action or 'rsp', latency_ms, in_tok, out_tok, False, retries)
                    return resp
                except Exception as e2:
                    _metrics.record(self.provider_name, model, action or 'rsp', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
                    raise e2
            _metrics.record(self.provider_name, model, action or 'rsp', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
            raise

    async def generate_embedding_async(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate embedding asynchronously with rate limiting and metrics"""
        in_tok = _estimate_tokens(text)
        
        # Run rate limiting in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._enforce_limits(model, in_tok))
        
        t0 = _now()
        retries = 0
        try:
            emb = await self.inner.generate_embedding_async(text, model, **kwargs)
            latency_ms = (_now() - t0) * 1000.0
            _metrics.record(self.provider_name, model, 'emb', latency_ms, in_tok, 0, False, retries)
            return emb
        except Exception as e:
            msg = str(e).lower()
            retry_after = 0.0
            if '429' in msg or 'rate limit' in msg:
                retry_after = 2.0
            if retry_after > 0 and retries < 2:
                retries += 1
                await asyncio.sleep(retry_after)  # Use async sleep
                try:
                    emb = await self.inner.generate_embedding_async(text, model, **kwargs)
                    latency_ms = (_now() - t0) * 1000.0
                    _metrics.record(self.provider_name, model, 'emb', latency_ms, in_tok, 0, False, retries)
                    return emb
                except Exception as e2:
                    _metrics.record(self.provider_name, model, 'emb', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
                    raise e2
            _metrics.record(self.provider_name, model, 'emb', (_now() - t0) * 1000.0, in_tok, 0, True, retries)
            raise
