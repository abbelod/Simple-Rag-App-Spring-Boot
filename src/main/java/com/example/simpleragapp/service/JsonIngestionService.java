package com.example.simpleragapp.service;


import org.springframework.ai.document.Document;
import org.springframework.ai.reader.JsonReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class JsonIngestionService {

    private final VectorStore vectorStore;
    private final TokenTextSplitter tokenTextSplitter;

    public JsonIngestionService(VectorStore vectorStore, TokenTextSplitter tokenTextSplitter) {
        this.vectorStore = vectorStore;
        this.tokenTextSplitter = tokenTextSplitter;
    }

    public int ingestJsonFilesFromFolder(String folderPath) throws IOException {

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("file:" + folderPath + "/*.json");

        if (resources.length ==  0) {
            return 0;
        }

        List<Document> allDocuments = new ArrayList<>();

        for (Resource resource : resources) {
            JsonReader reader = new JsonReader(resource);

            List<Document> documents= reader.get();
            allDocuments.addAll(documents);

        }

        List<Document> chunkedDocuments = tokenTextSplitter.apply(allDocuments);

        vectorStore.accept(chunkedDocuments);

        return chunkedDocuments.size();
    }

}
