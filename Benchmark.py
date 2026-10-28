"""Load test for a RUNNING server. Reports your REAL numbers.
  uvicorn app:app --workers 2        (terminal 1)
  python benchmark.py --n 2000 --concurrency 20   (terminal 2)
"""
import argparse, statistics, time, concurrent.futures as cf
import httpx

PAYLOAD = {"startingAirport": "JFK", "destinationAirport": "LAX",
           "isBasicEconomy": False, "isRefundable": True, "isNonStop": True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=20)
    a = ap.parse_args()
    lat, errors = [], 0

    def worker(k):
        nonlocal errors
        with httpx.Client(timeout=10) as c:
            for _ in range(k):
                t = time.perf_counter()
                try:
                    r = c.post(a.url, json=PAYLOAD)
                    r.raise_for_status()
                    lat.append((time.perf_counter() - t) * 1000)
                except Exception:
                    errors += 1

    per = a.n // a.concurrency
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(worker, [per] * a.concurrency))
    elapsed = time.perf_counter() - start
    lat.sort()
    q = lambda p: lat[int(len(lat) * p) - 1]
    print(f"requests ok: {len(lat)}, errors: {errors}")
    print(f"throughput: {len(lat)/elapsed:.0f} req/s")
    print(f"latency ms  p50={statistics.median(lat):.1f}  p95={q(.95):.1f}  p99={q(.99):.1f}")

if __name__ == "__main__":
    main()