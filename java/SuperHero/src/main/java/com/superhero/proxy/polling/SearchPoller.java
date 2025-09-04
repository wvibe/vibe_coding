package com.superhero.proxy.polling;

import com.fasterxml.jackson.databind.JsonNode;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.http.SuperHeroApiClient;
import com.superhero.proxy.v1.SearchResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class SearchPoller {
    private static final Logger log = LoggerFactory.getLogger(SearchPoller.class);

    private final SuperHeroApiClient apiClient;
    private final SearchCache cache;
    private final NameStore nameStore;
    private final long intervalSeconds;
    private final ExecutorService workers;
    private final ScheduledExecutorService scheduler;
    private final Map<String, String> lastHashByName = new ConcurrentHashMap<>();
    private final String pollingToken;

    public SearchPoller(SuperHeroApiClient apiClient,
                        SearchCache cache,
                        NameStore nameStore,
                        long intervalSeconds,
                        int maxConcurrency,
                        String pollingToken) {
        this.apiClient = apiClient;
        this.cache = cache;
        this.nameStore = nameStore;
        this.intervalSeconds = intervalSeconds;
        this.workers = Executors.newFixedThreadPool(Math.max(1, maxConcurrency));
        this.scheduler = Executors.newSingleThreadScheduledExecutor();
        this.pollingToken = pollingToken;
    }

    public void start() {
        log.info("poller starting with interval={}s", intervalSeconds);
        scheduler.scheduleAtFixedRate(this::runRoundSafe, intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
    }

    public void stop() {
        scheduler.shutdownNow();
        workers.shutdownNow();
    }

    private void runRoundSafe() {
        try {
            runRound();
        } catch (Throwable t) {
            log.warn("poll round failed", t);
        }
    }

    private void runRound() {
        Set<String> names = nameStore.view();
        int count = names.size();
        Instant start = Instant.now();
        log.info("poll-start names_count={}", count);

        if (count == 0) {
            log.info("poll-end names_count={} duration_ms={} fetched=0 updated=0 skipped_unchanged=0 errors=0",
                    count, Duration.between(start, Instant.now()).toMillis());
            return;
        }

        List<Runnable> tasks = new ArrayList<>();
        final AtomicInteger fetched = new AtomicInteger();
        final AtomicInteger updated = new AtomicInteger();
        final AtomicInteger skipped = new AtomicInteger();
        final AtomicInteger errors = new AtomicInteger();

        for (String name : names) {
            tasks.add(() -> {
                try {
                    if (pollingToken == null || pollingToken.isBlank()) {
                        skipped.incrementAndGet();
                        return;
                    }
                    JsonNode json = apiClient.search(pollingToken, name);
                    fetched.incrementAndGet();
                    SearchResponse resp = com.superhero.proxy.server.JsonToProtoMapper.toSearchResponse(json);
                    String newHash = hash(resp);
                    String oldHash = lastHashByName.put(name, newHash);
                    if (oldHash == null || !oldHash.equals(newHash)) {
                        cache.put(name, resp);
                        updated.incrementAndGet();
                    } else {
                        skipped.incrementAndGet();
                    }
                } catch (Exception e) {
                    errors.incrementAndGet();
                }
            });
        }

        // Shuffle to spread load if many
        Collections.shuffle(tasks);
        CountDownLatch latch = new CountDownLatch(tasks.size());
        for (Runnable task : tasks) {
            workers.submit(() -> {
                try {
                    task.run();
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await(Math.max(5L, Math.min(60L, intervalSeconds)), TimeUnit.SECONDS);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }

        long dur = Duration.between(start, Instant.now()).toMillis();
        log.info("poll-end names_count={} duration_ms={} fetched={} updated={} skipped_unchanged={} errors={}",
                count, dur, fetched.get(), updated.get(), skipped.get(), errors.get());
    }

    private static String hash(SearchResponse resp) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] bytes = resp.toByteArray();
            byte[] digest = md.digest(bytes);
            return Base64.getEncoder().encodeToString(digest);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}


