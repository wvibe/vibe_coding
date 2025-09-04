package com.superhero.proxy.polling;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.superhero.proxy.config.AppConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class FileBackedNameStore implements NameStore {
    private static final Logger log = LoggerFactory.getLogger(FileBackedNameStore.class);

    private final Path file;
    private final Set<String> names;
    private final ObjectMapper mapper = new ObjectMapper();

    public FileBackedNameStore(Path file) {
        this.file = file;
        this.names = ConcurrentHashMap.newKeySet();
    }

    @Override
    public Set<String> view() {
        return names;
    }

    private static String normalize(String name) {
        return name == null ? "" : name.trim().toLowerCase();
    }

    @Override
    public boolean add(String normalizedName) {
        if (normalizedName == null || normalizedName.isBlank()) return false;
        if (names.size() >= AppConfig.getNameSetMax()) {
            log.warn("name set is at capacity ({}). ignoring new name: {}", AppConfig.getNameSetMax(), normalizedName);
            return false;
        }
        return names.add(normalizedName);
    }

    @Override
    public void load() throws IOException {
        if (file == null) return;
        if (!Files.exists(file)) {
            log.info("names file does not exist: {}", file);
            return;
        }
        byte[] bytes = Files.readAllBytes(file);
        if (bytes.length == 0) return;
        List<String> list;
        try {
            list = mapper.readValue(bytes, new TypeReference<List<String>>(){});
        } catch (Exception e) {
            log.warn("failed to parse names file as JSON array, attempting line-based parse: {}", file);
            list = new ArrayList<>();
            for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                String s = line.trim();
                if (s.isEmpty() || s.startsWith("#")) continue;
                list.add(s);
            }
        }
        Set<String> normalized = new HashSet<>();
        for (String s : list) {
            String n = normalize(s);
            if (!n.isBlank()) normalized.add(n);
        }
        names.clear();
        names.addAll(normalized);
        log.info("loaded {} names from {}", names.size(), file);
    }

    @Override
    public void save() throws IOException {
        if (file == null) return;
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName().toString() + ".tmp");
        byte[] content = mapper.writeValueAsBytes(new ArrayList<>(names));
        Files.write(tmp, content);
        Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        log.info("saved {} names to {} ({} bytes)", names.size(), file, content.length);
    }
}


