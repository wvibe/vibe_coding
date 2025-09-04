package com.superhero.proxy.polling;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.superhero.proxy.cache.CaffeineSearchCache;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.http.SuperHeroApiClient;
import com.superhero.proxy.v1.SearchResponse;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class SearchPollerTest {

    static class FakeApiClient extends SuperHeroApiClient {
        private final ObjectMapper mapper = new ObjectMapper();
        private final String body;
        public FakeApiClient(String body) {
            super("http://unused");
            this.body = body;
        }
        @Override
        public JsonNode search(String accessToken, String name) {
            try { return mapper.readTree(body); } catch (Exception e) { throw new RuntimeException(e); }
        }
    }

    @Test
    void poller_updates_cache_on_change() throws Exception {
        Path tmp = Files.createTempFile("names", ".json");
        FileBackedNameStore store = new FileBackedNameStore(tmp);
        store.add("batman");

        String successJson = "{\"response\":\"success\",\"results-for\":\"batman\",\"results\":[{\"id\":\"70\",\"name\":\"Batman\"}]}";
        SuperHeroApiClient api = new FakeApiClient(successJson);
        SearchCache cache = new CaffeineSearchCache();
        SearchPoller poller = new SearchPoller(api, cache, store, 1, 1, "t");
        poller.start();

        Thread.sleep(1500);
        assertTrue(cache.get("batman").isPresent());
        SearchResponse resp = cache.get("batman").get();
        assertEquals("success", resp.getResponse());

        poller.stop();
    }
}


