"""Small network helpers."""
import httpx

# Browser-like User-Agent: some sites block or throttle the default httpx one.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


async def url_status(url: str, cache: dict[str, int]) -> int:
    """HTTP status after redirects; 0 when unreachable. Cached per job, shared by check_inputs and QC.

    Args:
        url: The URL to probe.
        cache: Mutable ``url -> status`` map owned by the caller (one per job). Read first and updated in place,
            so each URL is requested at most once per job.

    Returns:
        The final HTTP status code, or ``0`` if the request failed (DNS, timeout, connection error, ...).
    """
    if url in cache:
        return cache[url]
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8, headers={"User-Agent": UA}) as c:
            # HEAD is cheaper than GET since no body is downloaded.
            r = await c.head(url)
            if r.status_code in (403, 405, 501):          # some sites refuse HEAD but allow GET
                r = await c.get(url)
            cache[url] = r.status_code
    except httpx.HTTPError:
        # Network-level failure: record as unreachable so we don't retry within the same job.
        cache[url] = 0
    return cache[url]
