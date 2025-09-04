package com.superhero.proxy.polling;

import java.io.IOException;
import java.util.Set;

public interface NameStore {
    Set<String> view();
    boolean add(String normalizedName);
    void load() throws IOException;
    void save() throws IOException;
}


