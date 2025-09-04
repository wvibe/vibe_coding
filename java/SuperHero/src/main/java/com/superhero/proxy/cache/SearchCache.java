package com.superhero.proxy.cache;

import com.superhero.proxy.v1.SearchResponse;

import java.util.Optional;

public interface SearchCache {
    Optional<SearchResponse> get(String normalizedName);
    void put(String normalizedName, SearchResponse response);
    void invalidate(String normalizedName);
    void clear();

    long getCacheHits();
    long getCacheMisses();
}


