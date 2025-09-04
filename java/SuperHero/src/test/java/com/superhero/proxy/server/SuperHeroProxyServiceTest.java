package com.superhero.proxy.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.superhero.proxy.cache.CaffeineSearchCache;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.http.SuperHeroApiClient;
import com.superhero.proxy.v1.SearchRequest;
import com.superhero.proxy.v1.SearchResponse;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

public class SuperHeroProxyServiceTest {

    static class FakeApiClient extends SuperHeroApiClient {
        private final ObjectMapper mapper = new ObjectMapper();
        private final AtomicInteger callCount = new AtomicInteger();
        private final boolean success;

        public FakeApiClient(boolean success) {
            super("http://unused");
            this.success = success;
        }

        @Override
        public JsonNode search(String accessToken, String name) {
            callCount.incrementAndGet();
            try {
                if (success) {
                    String json = "{\n" +
                            "  \"response\": \"success\",\n" +
                            "  \"results-for\": \"batman\",\n" +
                            "  \"results\": [{\n" +
                            "    \"id\": \"70\",\n" +
                            "    \"name\": \"Batman\"\n" +
                            "  }]\n" +
                            "}";
                    return mapper.readTree(json);
                } else {
                    String json = "{\n" +
                            "  \"response\": \"error\",\n" +
                            "  \"error\": \"character with given name not found\"\n" +
                            "}";
                    return mapper.readTree(json);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        int getCallCount() { return callCount.get(); }
    }

    static class TestObserver implements StreamObserver<SearchResponse> {
        SearchResponse response;
        Throwable error;

        @Override
        public void onNext(SearchResponse value) { this.response = value; }
        @Override
        public void onError(Throwable t) { this.error = t; }
        @Override
        public void onCompleted() { }
    }

    @Test
    void cache_miss_then_hit_success() {
        FakeApiClient api = new FakeApiClient(true);
        SearchCache cache = new CaffeineSearchCache();
        SuperHeroProxyService service = new SuperHeroProxyService(api, cache);

        SearchRequest req = SearchRequest.newBuilder()
                .setName("batman")
                .setToken("t")
                .build();

        TestObserver obs1 = new TestObserver();
        service.searchHeroes(req, obs1);
        assertNull(obs1.error);
        assertNotNull(obs1.response);
        assertEquals(1, api.getCallCount());
        long missesAfterFirst = cache.getCacheMisses();
        assertEquals(1, missesAfterFirst);

        TestObserver obs2 = new TestObserver();
        service.searchHeroes(req, obs2);
        assertNull(obs2.error);
        assertNotNull(obs2.response);
        assertEquals(1, api.getCallCount(), "second call should be served from cache");
        assertTrue(cache.getCacheHits() >= 1);
    }

    @Test
    void error_is_not_cached() {
        FakeApiClient api = new FakeApiClient(false);
        SearchCache cache = new CaffeineSearchCache();
        SuperHeroProxyService service = new SuperHeroProxyService(api, cache);

        SearchRequest req = SearchRequest.newBuilder()
                .setName("unknown")
                .setToken("t")
                .build();

        TestObserver obs1 = new TestObserver();
        service.searchHeroes(req, obs1);
        assertNull(obs1.response);
        assertNotNull(obs1.error);
        assertTrue(obs1.error instanceof StatusRuntimeException);
        assertEquals(1, api.getCallCount());
        long misses1 = cache.getCacheMisses();
        assertEquals(1, misses1);

        TestObserver obs2 = new TestObserver();
        service.searchHeroes(req, obs2);
        assertNull(obs2.response);
        assertNotNull(obs2.error);
        assertEquals(2, api.getCallCount(), "should call API again because error not cached");
        assertEquals(2, cache.getCacheMisses());
        assertEquals(0, cache.getCacheHits());
    }
}


