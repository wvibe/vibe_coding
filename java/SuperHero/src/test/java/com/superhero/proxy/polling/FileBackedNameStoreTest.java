package com.superhero.proxy.polling;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class FileBackedNameStoreTest {

    @Test
    void load_json_array_and_normalize() throws Exception {
        Path tmp = Files.createTempFile("names", ".json");
        Files.writeString(tmp, "[\n  \" Batman \", \"superman\", \"SPIDER-MAN\"\n]", StandardCharsets.UTF_8);

        FileBackedNameStore store = new FileBackedNameStore(tmp);
        store.load();

        assertTrue(store.view().contains("batman"));
        assertTrue(store.view().contains("superman"));
        assertTrue(store.view().contains("spider-man"));
        assertEquals(3, store.view().size());
    }

    @Test
    void save_writes_atomic_and_roundtrip() throws Exception {
        Path tmp = Files.createTempFile("names", ".json");
        FileBackedNameStore store = new FileBackedNameStore(tmp);
        store.add("batman");
        store.add("superman");
        store.save();

        String content = Files.readString(tmp, StandardCharsets.UTF_8);
        assertTrue(content.contains("batman"));
        assertTrue(content.contains("superman"));

        // Reload into a new instance and assert
        FileBackedNameStore store2 = new FileBackedNameStore(tmp);
        store2.load();
        assertEquals(store.view(), store2.view());
    }
}


