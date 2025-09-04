package com.superhero.proxy.server;

import com.superhero.proxy.config.AppConfig;
import com.superhero.proxy.cache.CaffeineSearchCache;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.http.SuperHeroApiClient;
import com.superhero.proxy.v1.SearchRequest;
import com.superhero.proxy.v1.SearchResponse;
import com.superhero.proxy.v1.SuperHeroProxyGrpc;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.Server;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class SuperHeroProxyServiceIT {

    private static Server server;
    private static int port;
    private static ManagedChannel channel;

    @BeforeAll
    static void startServer() throws IOException {
        SuperHeroApiClient apiClient = new SuperHeroApiClient(AppConfig.getSuperHeroApiBaseUrl());
        SearchCache cache = new CaffeineSearchCache();
        server = NettyServerBuilder.forPort(0)
                .addService(new SuperHeroProxyService(apiClient, cache))
                .build()
                .start();
        port = server.getPort();
        channel = NettyChannelBuilder.forAddress("localhost", port)
                .usePlaintext()
                .build();
    }

    @AfterAll
    static void stopServer() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @Order(1)
    void searchBatman_success() {
        String token = System.getenv("SUPERHERO_TEST_TOKEN");
        if (token == null || token.isBlank()) {
            token = System.getProperty("superhero.test.token");
        }
        assumeTrue(token != null && !token.isBlank(), "Set SUPERHERO_TEST_TOKEN env or -Dsuperhero.test.token");

        SuperHeroProxyGrpc.SuperHeroProxyBlockingStub stub = SuperHeroProxyGrpc.newBlockingStub(channel);
        SearchRequest req = SearchRequest.newBuilder()
                .setName("batman")
                .setToken(token)
                .build();

        SearchResponse resp = stub.searchHeroes(req);
        assertNotNull(resp);
        assertEquals("success", resp.getResponse());
        assertEquals("batman", resp.getResultsFor().toLowerCase());
        assertTrue(resp.getResultsCount() > 0, "Expected at least one result for batman");

        // Call again to ensure served quickly (cache hit). We can't assert time reliably,
        // but ensuring no error and same results count indicates consistent behavior.
        SearchResponse resp2 = stub.searchHeroes(req);
        assertNotNull(resp2);
        assertEquals("success", resp2.getResponse());
        assertTrue(resp2.getResultsCount() >= resp.getResultsCount());
    }
}


