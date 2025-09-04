SuperHero gRPC Proxy (Java 21 + Maven)

Prerequisites
- Java 21
- Maven 3.9+

Environment
- GRPC_PORT (optional, default 9090)
- SUPERHERO_API_BASEURL (optional, default https://superheroapi.com/api)
- CACHE_TTL_SECONDS (optional, default 60)
- CACHE_MAX_SIZE (optional, default 1000)

Build and Run
```bash
cd /Users/weimu/Development/vibe/vibe_coding/java/SuperHero
mvn -q -DskipTests package
mvn -q exec:java -Dexec.mainClass=com.superhero.proxy.server.GrpcServer
```

Proto
- src/main/proto/superhero_proxy.proto defines the SearchHeroes RPC.

Test (basic)
Use a gRPC client of your choice to invoke `com.superhero.proxy.v1.SuperHeroProxy/SearchHeroes` with body:
```json
{ "name": "batman", "token": "<your_access_token>" }
```

References
- SuperHero API: https://superheroapi.com/

Cache
- Key: normalized name only (trim + lower-case)
- Value: full SearchResponse (protobuf)
- Policy: TTL 1 minute, max size 1000, success responses only

