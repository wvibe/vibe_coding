# SuperHero gRPC Proxy — Design Notes

Reference: https://superheroapi.com/

## Overview
This project implements a gRPC proxy to the SuperHero REST API, a local in-memory cache to improve latency, and an optional asynchronous poller that refreshes cached results and persists a list of names to watch.

Java 21 + Maven; gRPC-Java; Protobuf; Jackson for JSON; Caffeine for caching; JDK HttpClient for REST; SLF4J/Logback for logging.

## Part 1: gRPC Proxy
- RPC: `SuperHeroProxy.SearchHeroes(SearchRequest) → SearchResponse`
  - `SearchRequest { string name; string token; }`
  - `SearchResponse` mirrors REST fields (`response`, `results-for`, `results[...]`).
- Token handling: The access token is supplied by the client in the request (no secrets in code).
- HTTP client: JDK `java.net.http.HttpClient` with timeouts and redirects enabled.
- JSON mapping: Jackson parses the SuperHero API JSON; hyphenated keys mapped to snake_case in proto messages.
- Error handling:
  - Missing name/token → gRPC `INVALID_ARGUMENT`.
  - Network/HTTP errors (non-200/IO) → `UNAVAILABLE`.
  - Upstream `{"response":"error"}` → `NOT_FOUND` with upstream message.
- Server reflection: Enabled to support grpcurl/discovery.
- Configuration (env):
  - `GRPC_PORT` (default 9090)
  - `SUPERHERO_API_BASEURL` (default `https://superheroapi.com/api`)

## Part 2: Caching
- Library: Caffeine in-memory cache.
- Key: normalized search name only (trim + lower-case). Cross-user by design.
- Value: full `SearchResponse` protobuf (immutable, safe to share).
- Policy: cache only successful responses; TTL 1 minute; max size 1000.
- Metrics: simple hit/miss counters inside the cache implementation.
- Configuration (env):
  - `CACHE_TTL_SECONDS` (default 60)
  - `CACHE_MAX_SIZE` (default 1000)
- Integration path:
  1) On request, check cache by normalized name.
  2) Cache hit → return immediately.
  3) Cache miss → call REST, map to proto, store on success, return.

## Part 3: Async Polling (no streaming)
- Goal: Keep cache fresh and maintain a persisted list of names to watch.
- Name list management:
  - In-memory: thread-safe `Set<String>` of normalized names.
  - Persistence: file-backed JSON array via `FileBackedNameStore`.
    - Load once at server start.
    - Periodic atomic save (tmp + atomic move) and on shutdown.
  - Additions: Only added when `SearchHeroes` returns upstream `success` with at least one result.
  - No live reload: runtime edits to the file are not read; the saver will overwrite the file with the current in-memory set.
- Poller:
  - Scheduled at `UPDATE_POLL_INTERVAL_SECONDS`.
  - Reads current in-memory names; fetches via `SuperHeroApiClient` using `POLLING_SERVICE_TOKEN`.
  - Change detection: compute SHA-256 hash of `SearchResponse.toByteArray()`; on change, update cache and last-hash map.
  - Concurrency: fixed worker pool (default 4). Per-round stats logged: `fetched`, `updated`, `skipped_unchanged`, `errors`.
- Configuration (env):
  - `POLLING_ENABLED` (default false)
  - `POLLING_NAMES_FILE` (required when enabled; JSON array path)
  - `POLLING_SERVICE_TOKEN` (token used by poller to call REST)
  - `UPDATE_POLL_INTERVAL_SECONDS` (default 60)
  - `UPDATE_MAX_CONCURRENCY` (default 4)
  - `POLLING_SAVE_INTERVAL_SECONDS` (default 30)
  - `NAME_SET_MAX` (default 10000)

## Code Structure
- `src/main/proto/superhero_proxy.proto`: Protobuf schema; service and messages.
- `com.superhero.proxy.server`
  - `GrpcServer`: boots the gRPC server, reflection, and (optionally) the poller and periodic saver.
  - `SuperHeroProxyService`: gRPC implementation; validates input, checks cache, calls REST, maps response, caches, conditionally ingests name.
  - `JsonToProtoMapper`: JSON → protobuf mapping (shared with poller).
- `com.superhero.proxy.http`
  - `SuperHeroApiClient`: REST client (JDK HttpClient + redirects + timeouts).
- `com.superhero.proxy.cache`
  - `SearchCache` (interface), `CaffeineSearchCache` (TTL + size + hit/miss).
- `com.superhero.proxy.polling`
  - `NameStore`, `FileBackedNameStore`: in-memory set + atomic file persistence.
  - `SearchPoller`: scheduled fetcher that updates cache on changes and logs per-round stats.
- `com.superhero.proxy.config`
  - `AppConfig`: environment-backed configuration getters.

## Run and Test
- Start server:
```
GRPC_PORT=9090 \
POLLING_ENABLED=true \
POLLING_NAMES_FILE=/tmp/superhero_names.json \
POLLING_SERVICE_TOKEN=<token> \
UPDATE_POLL_INTERVAL_SECONDS=60 \
POLLING_SAVE_INTERVAL_SECONDS=30 \
MAVEN_OPTS="" mvn -q exec:java -Dexec.mainClass=com.superhero.proxy.server.GrpcServer
```
- Call RPC via grpcurl:
```
grpcurl -plaintext -d '{"name":"batman","token":"<token>"}' \
  localhost:9090 com.superhero.proxy.v1.SuperHeroProxy/SearchHeroes
```
- Tests:
```
mvn test
# With live integration test token:
mvn -Dsuperhero.test.token=<token> test
```

## Notes / Trade-offs
- Name key is search-string only to promote sharing across users; if isolation is needed, include token as part of the key.
- Poller currently focuses on search results; by-id update flows or streaming can be added later if needed.
- Persistence format is a simple JSON array for human-friendliness and easy bootstrapping.
