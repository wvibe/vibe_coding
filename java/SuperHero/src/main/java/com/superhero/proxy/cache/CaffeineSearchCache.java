package com.superhero.proxy.cache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.superhero.proxy.config.AppConfig;
import com.superhero.proxy.v1.SearchResponse;

import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

public class CaffeineSearchCache implements SearchCache {
    private final Cache<String, SearchResponse> cache;
    private final AtomicLong cacheHits = new AtomicLong();
    private final AtomicLong cacheMisses = new AtomicLong();

    public CaffeineSearchCache() {
        long ttlSeconds = AppConfig.getCacheTtlSeconds();
        long maxSize = AppConfig.getCacheMaxSize();
        this.cache = Caffeine.newBuilder()
                .expireAfterWrite(Duration.ofSeconds(ttlSeconds))
                .maximumSize(maxSize)
                .build();
    }

    @Override
    public Optional<SearchResponse> get(String normalizedName) {
        SearchResponse value = cache.getIfPresent(normalizedName);
        if (value != null) {
            cacheHits.incrementAndGet();
            return Optional.of(value);
        }
        cacheMisses.incrementAndGet();
        return Optional.empty();
    }

    @Override
    public void put(String normalizedName, SearchResponse response) {
        cache.put(normalizedName, response);
    }

    @Override
    public void invalidate(String normalizedName) {
        cache.invalidate(normalizedName);
    }

    @Override
    public void clear() {
        cache.invalidateAll();
    }

    @Override
    public long getCacheHits() {
        return cacheHits.get();
    }

    @Override
    public long getCacheMisses() {
        return cacheMisses.get();
    }
}


