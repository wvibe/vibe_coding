package com.superhero.proxy.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.superhero.proxy.cache.CaffeineSearchCache;
import com.superhero.proxy.cache.SearchCache;
import com.superhero.proxy.polling.NameStore;
import com.superhero.proxy.http.SuperHeroApiClient;
import com.superhero.proxy.v1.Appearance;
import com.superhero.proxy.v1.Biography;
import com.superhero.proxy.v1.Connections;
import com.superhero.proxy.v1.Hero;
import com.superhero.proxy.v1.Image;
import com.superhero.proxy.v1.Powerstats;
import com.superhero.proxy.v1.SearchRequest;
import com.superhero.proxy.v1.SearchResponse;
import com.superhero.proxy.v1.SuperHeroProxyGrpc;
import com.superhero.proxy.v1.Work;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class SuperHeroProxyService extends SuperHeroProxyGrpc.SuperHeroProxyImplBase {
    private static final Logger log = LoggerFactory.getLogger(SuperHeroProxyService.class);

    private final SuperHeroApiClient apiClient;
    private final SearchCache cache;
    private final NameStore nameStore; // optional, may be null when polling disabled
    private final java.util.concurrent.ExecutorService nameIngestExecutor = java.util.concurrent.Executors.newSingleThreadExecutor();

    public SuperHeroProxyService(SuperHeroApiClient apiClient) {
        this.apiClient = apiClient;
        this.cache = new CaffeineSearchCache();
        this.nameStore = null;
    }

    public SuperHeroProxyService(SuperHeroApiClient apiClient, SearchCache cache) {
        this.apiClient = apiClient;
        this.cache = cache;
        this.nameStore = null;
    }

    public SuperHeroProxyService(SuperHeroApiClient apiClient, SearchCache cache, NameStore nameStore) {
        this.apiClient = apiClient;
        this.cache = cache;
        this.nameStore = nameStore;
    }

    @Override
    public void searchHeroes(SearchRequest request, StreamObserver<SearchResponse> responseObserver) {
        String name = request.getName();
        String token = request.getToken();
        if (name == null || name.isBlank()) {
            responseObserver.onError(Status.INVALID_ARGUMENT.withDescription("name is required").asRuntimeException());
            return;
        }
        if (token == null || token.isBlank()) {
            responseObserver.onError(Status.INVALID_ARGUMENT.withDescription("token is required").asRuntimeException());
            return;
        }

        try {
            String normalizedName = name.trim().toLowerCase();

            // Cache check
            var cached = cache.get(normalizedName);
            if (cached.isPresent()) {
                responseObserver.onNext(cached.get());
                responseObserver.onCompleted();
                return;
            }

            JsonNode json = apiClient.search(token, normalizedName);
            String apiResponse = json.path("response").asText("");
            if ("error".equalsIgnoreCase(apiResponse)) {
                String errorMsg = json.path("error").asText("unknown error");
                responseObserver.onError(Status.NOT_FOUND.withDescription(errorMsg).asRuntimeException());
                return;
            }

            SearchResponse.Builder builder = SearchResponse.newBuilder()
                    .setResponse(apiResponse)
                    .setResultsFor(json.path("results-for").asText(""));

            JsonNode results = json.path("results");
            if (results.isArray()) {
                for (JsonNode heroNode : results) {
                    builder.addResults(mapHero(heroNode));
                }
            }

            SearchResponse built = builder.build();
            cache.put(normalizedName, built);

            // Only add to polling name list if upstream responded successfully and returned results
            if (nameStore != null && "success".equalsIgnoreCase(apiResponse) && built.getResultsCount() > 0) {
                nameIngestExecutor.submit(() -> nameStore.add(normalizedName));
            }
            responseObserver.onNext(built);
            responseObserver.onCompleted();
        } catch (IOException e) {
            log.error("I/O error calling SuperHero API", e);
            responseObserver.onError(Status.UNAVAILABLE.withDescription("I/O error calling SuperHero API").withCause(e).asRuntimeException());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            responseObserver.onError(Status.CANCELLED.withDescription("Interrupted").asRuntimeException());
        } catch (Exception e) {
            log.error("Unexpected error", e);
            responseObserver.onError(Status.UNKNOWN.withDescription("Unexpected error").withCause(e).asRuntimeException());
        }
    }

    private static Hero mapHero(JsonNode node) {
        Hero.Builder hero = Hero.newBuilder();
        hero.setId(node.path("id").asText(""));
        hero.setName(node.path("name").asText(""));

        JsonNode powerstats = node.path("powerstats");
        Powerstats.Builder ps = Powerstats.newBuilder()
                .setIntelligence(powerstats.path("intelligence").asText(""))
                .setStrength(powerstats.path("strength").asText(""))
                .setSpeed(powerstats.path("speed").asText(""))
                .setDurability(powerstats.path("durability").asText(""))
                .setPower(powerstats.path("power").asText(""))
                .setCombat(powerstats.path("combat").asText(""));
        hero.setPowerstats(ps.build());

        JsonNode biography = node.path("biography");
        Biography.Builder bio = Biography.newBuilder()
                .setFullName(biography.path("full-name").asText(""))
                .setAlterEgos(biography.path("alter-egos").asText(""))
                .setPlaceOfBirth(biography.path("place-of-birth").asText(""))
                .setFirstAppearance(biography.path("first-appearance").asText(""))
                .setPublisher(biography.path("publisher").asText(""))
                .setAlignment(biography.path("alignment").asText(""));
        JsonNode aliases = biography.path("aliases");
        if (aliases.isArray()) {
            for (JsonNode alias : aliases) {
                bio.addAliases(alias.asText(""));
            }
        }
        hero.setBiography(bio.build());

        JsonNode appearance = node.path("appearance");
        Appearance.Builder app = Appearance.newBuilder()
                .setGender(appearance.path("gender").asText(""))
                .setRace(appearance.path("race").asText(""))
                .setEyeColor(appearance.path("eye-color").asText(""))
                .setHairColor(appearance.path("hair-color").asText(""));
        JsonNode height = appearance.path("height");
        if (height.isArray()) {
            for (JsonNode h : height) {
                app.addHeight(h.asText(""));
            }
        }
        JsonNode weight = appearance.path("weight");
        if (weight.isArray()) {
            for (JsonNode w : weight) {
                app.addWeight(w.asText(""));
            }
        }
        hero.setAppearance(app.build());

        JsonNode work = node.path("work");
        Work.Builder workB = Work.newBuilder()
                .setOccupation(work.path("occupation").asText(""))
                .setBase(work.path("base").asText(""));
        hero.setWork(workB.build());

        JsonNode connections = node.path("connections");
        Connections.Builder conn = Connections.newBuilder()
                .setGroupAffiliation(connections.path("group-affiliation").asText(""))
                .setRelatives(connections.path("relatives").asText(""));
        hero.setConnections(conn.build());

        JsonNode image = node.path("image");
        Image.Builder img = Image.newBuilder().setUrl(image.path("url").asText(""));
        hero.setImage(img.build());

        return hero.build();
    }
}


