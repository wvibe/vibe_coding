package com.superhero.proxy.config;

import java.util.Optional;

public final class AppConfig {

    private static final String DEFAULT_BASE_URL = "https://superheroapi.com/api";
    private static final int DEFAULT_GRPC_PORT = 9090;
    private static final long DEFAULT_CACHE_TTL_SECONDS = 60L;
    private static final long DEFAULT_CACHE_MAX_SIZE = 1000L;
    private static final boolean DEFAULT_POLLING_ENABLED = false;
    private static final long DEFAULT_POLL_INTERVAL_SECONDS = 60L;
    private static final int DEFAULT_POLL_MAX_CONCURRENCY = 4;
    private static final long DEFAULT_POLL_SAVE_INTERVAL_SECONDS = 30L;
    private static final int DEFAULT_NAME_SET_MAX = 10000;

    private AppConfig() {}

    public static String getSuperHeroApiToken() {
        return System.getenv("SUPERHERO_API_TOKEN");
    }

    public static String getSuperHeroApiBaseUrl() {
        return Optional.ofNullable(System.getenv("SUPERHERO_API_BASEURL")).orElse(DEFAULT_BASE_URL);
    }

    public static int getGrpcPort() {
        String raw = System.getenv("GRPC_PORT");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_GRPC_PORT;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_GRPC_PORT;
        }
    }

    public static long getCacheTtlSeconds() {
        String raw = System.getenv("CACHE_TTL_SECONDS");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_CACHE_TTL_SECONDS;
        }
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_CACHE_TTL_SECONDS;
        }
    }

    public static long getCacheMaxSize() {
        String raw = System.getenv("CACHE_MAX_SIZE");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_CACHE_MAX_SIZE;
        }
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_CACHE_MAX_SIZE;
        }
    }

    public static boolean isPollingEnabled() {
        String raw = System.getenv("POLLING_ENABLED");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_POLLING_ENABLED;
        }
        return raw.equalsIgnoreCase("1") || raw.equalsIgnoreCase("true") || raw.equalsIgnoreCase("yes");
    }

    public static String getPollingNamesFile() {
        return System.getenv("POLLING_NAMES_FILE");
    }

    public static long getUpdatePollIntervalSeconds() {
        String raw = System.getenv("UPDATE_POLL_INTERVAL_SECONDS");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_POLL_INTERVAL_SECONDS;
        }
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_POLL_INTERVAL_SECONDS;
        }
    }

    public static int getUpdateMaxConcurrency() {
        String raw = System.getenv("UPDATE_MAX_CONCURRENCY");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_POLL_MAX_CONCURRENCY;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_POLL_MAX_CONCURRENCY;
        }
    }

    public static long getPollingSaveIntervalSeconds() {
        String raw = System.getenv("POLLING_SAVE_INTERVAL_SECONDS");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_POLL_SAVE_INTERVAL_SECONDS;
        }
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_POLL_SAVE_INTERVAL_SECONDS;
        }
    }

    public static int getNameSetMax() {
        String raw = System.getenv("NAME_SET_MAX");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_NAME_SET_MAX;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            return DEFAULT_NAME_SET_MAX;
        }
    }
}


