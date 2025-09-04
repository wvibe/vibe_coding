package com.superhero.proxy.server;

import com.superhero.proxy.config.AppConfig;
import com.superhero.proxy.cache.CaffeineSearchCache;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.polling.FileBackedNameStore;
import com.superhero.proxy.polling.NameStore;
import com.superhero.proxy.polling.SearchPoller;
import com.superhero.proxy.http.SuperHeroApiClient;
import io.grpc.Server;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.services.ProtoReflectionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class GrpcServer {
    private static final Logger log = LoggerFactory.getLogger(GrpcServer.class);

    private Server server;
    private SearchPoller poller;
    private java.util.concurrent.ScheduledExecutorService saverScheduler;
    private NameStore nameStoreField;

    private void start() throws IOException {
        int port = AppConfig.getGrpcPort();
        String baseUrl = AppConfig.getSuperHeroApiBaseUrl();
        SuperHeroApiClient apiClient = new SuperHeroApiClient(baseUrl);
        SearchCache cache = new CaffeineSearchCache();
        NameStore nameStore = null;
        if (AppConfig.isPollingEnabled()) {
            String file = AppConfig.getPollingNamesFile();
            if (file != null && !file.isBlank()) {
                nameStore = new FileBackedNameStore(java.nio.file.Path.of(file));
                try { nameStore.load(); } catch (Exception e) { log.warn("Failed to load names file", e); }
            } else {
                log.warn("POLLING_ENABLED is true but POLLING_NAMES_FILE is not set; poller will have no names");
            }
        }
        this.nameStoreField = nameStore;

        this.server = NettyServerBuilder.forPort(port)
                .addService(new SuperHeroProxyService(apiClient, cache, nameStoreField))
                .addService(ProtoReflectionService.newInstance())
                .build()
                .start();

        log.info("gRPC server started, listening on port {}", port);

        if (AppConfig.isPollingEnabled() && nameStoreField != null) {
            this.poller = new SearchPoller(
                    apiClient,
                    cache,
                    nameStoreField,
                    AppConfig.getUpdatePollIntervalSeconds(),
                    AppConfig.getUpdateMaxConcurrency(),
                    System.getenv("POLLING_SERVICE_TOKEN")
            );
            this.poller.start();

            this.saverScheduler = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
            this.saverScheduler.scheduleAtFixedRate(() -> {
                try { nameStoreField.save(); } catch (Exception e) { log.warn("Failed to save names file", e); }
            }, AppConfig.getPollingSaveIntervalSeconds(), AppConfig.getPollingSaveIntervalSeconds(), java.util.concurrent.TimeUnit.SECONDS);
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            log.info("JVM shutdown detected. Shutting down gRPC server...");
            try {
                if (saverScheduler != null) {
                    saverScheduler.shutdownNow();
                }
                if (poller != null) {
                    poller.stop();
                }
                // final save
                try {
                    if (nameStoreField != null) {
                        nameStoreField.save();
                    }
                } catch (Exception e) { log.warn("final names save failed", e); }
                GrpcServer.this.stop();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            log.info("Server stopped.");
        }));
    }

    private void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final GrpcServer server = new GrpcServer();
        server.start();
        server.blockUntilShutdown();
    }
}


