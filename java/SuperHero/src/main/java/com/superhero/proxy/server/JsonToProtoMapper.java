package com.superhero.proxy.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.superhero.proxy.v1.*;

public final class JsonToProtoMapper {
    private JsonToProtoMapper() {}

    public static SearchResponse toSearchResponse(JsonNode json) {
        String apiResponse = json.path("response").asText("");
        SearchResponse.Builder builder = SearchResponse.newBuilder()
                .setResponse(apiResponse)
                .setResultsFor(json.path("results-for").asText(""));
        JsonNode results = json.path("results");
        if (results.isArray()) {
            for (JsonNode heroNode : results) {
                builder.addResults(toHero(heroNode));
            }
        }
        return builder.build();
    }

    public static Hero toHero(JsonNode node) {
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


