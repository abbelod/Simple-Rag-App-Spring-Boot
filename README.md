# Simple RAG using Spring Boot

This guide walks through setting up a lightweight, hybrid Retrieval-Augmented Generation (RAG) system using **Spring Boot**. We will use **Ollama** locally for free vector embeddings and **Groq** for high-speed AI chat responses.

---

## Prerequisites & Setup

1. **Install Ollama** locally 
``` 
curl -fsSL https://ollama.com/install.sh | sh
```
pull the embedding model:
```bash
ollama pull nomic-embed-text

```


2. **Get a Groq API Key** from [Groq Console](https://console.groq.com/keys).

---

## Project Structure & File Placeholders


### 1. Maven Dependencies
We need the following dependencies
> **File:** `pom.xml`

```xml
<!-- PASTE YOUR pom.xml DEPENDENCIES HERE -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webmvc</artifactId>
</dependency>

<dependency>
<groupId>org.springframework.ai</groupId>
<artifactId>spring-ai-vector-store</artifactId>
</dependency>

<dependency>
<groupId>org.springframework.ai</groupId>
<artifactId>spring-ai-tika-document-reader</artifactId>
</dependency>

<dependency>
<groupId>org.springframework.ai</groupId>
<artifactId>spring-ai-vector-store-advisor</artifactId>
</dependency>

<dependency>
<groupId>org.springframework.ai</groupId>
<artifactId>spring-ai-starter-model-openai</artifactId>
</dependency>

```

---

### 2. Application Configuration
Store the GROQ API key in an environment variable and reference it in your application.yml file.
We will be using the llama-3.3-70b-versatile model provided.
> **File:** `src/main/resources/application.yml`

```yaml
# PASTE YOUR application.yml CONFIGURATION HERE
spring:
  application:
    name: Simple-Rag-App
  ai:
    openai:
      chat:
        base-url: https://api.groq.com/openai/v1
        model: llama-3.3-70b-versatile
        api-key: ${OPENAI_API_KEY}
      embedding:
        base-url: http://localhost:11434/v1
        model: nomic-embed-text
```

---

### 3. Vector Store Configuration

> **File:** `src/main/java/com/example/ragdemo/config/RagConfig.java`

```java
package com.example.simpleragapp.config;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
@Configuration
public class AiConfig {

    @Bean
    public VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }
    
    @Bean
    public ChatClient chatClient(ChatClient.Builder builder) {
        return builder.build();
    }
}
```

---

### 4. REST Controller

> **File:** `src/main/java/com/example/ragdemo/controller/RagController.java`

```java
package com.example.simpleragapp.controller;
import com.example.simpleragapp.dto.SearchRequestPayload;
import com.example.simpleragapp.dto.SearchResponse;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/api/v1/rag")
public class RagController {

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    public RagController(VectorStore vectorStore, ChatClient chatClient) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
    }

    @PostMapping("/upload")
    public ResponseEntity<Map<String, String>> uploadDocument(@RequestParam("file")MultipartFile file) {
        if(file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("message", "File cannot be uploaded"));
        }

        try {
            InputStreamResource resource = new InputStreamResource(file.getInputStream());
            TikaDocumentReader documentReader = new TikaDocumentReader(resource);
            List<Document> documents = documentReader.read();

            TokenTextSplitter textSplitter = TokenTextSplitter.builder()
                    .withChunkSize(800)
                    .withMinChunkSizeChars(350)
                    .withMinChunkLengthToEmbed(5)
                    .withMaxNumChunks(1000)
                    .withKeepSeparator(true)
                    .build();

            List<Document> chunkedDocuments = textSplitter.apply(documents);
            
            vectorStore.accept(chunkedDocuments);

            return ResponseEntity.ok(Map.of(
                    "status", "success",
                    "filename", file.getOriginalFilename(),
                    "chunksIngested", String.valueOf(chunkedDocuments.size())
            ));
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of("error", "Failed to process document: "+ e.getMessage()));
        }
    }
    
    @GetMapping("/search")
    public ResponseEntity<SearchResponse> searchAndAnswer(@RequestParam("query") SearchRequestPayload payload) {
        if(payload.query() == null  || payload.query().isBlank()) {
            return ResponseEntity.badRequest().build();
        }

        String answer = chatClient.prompt()
                .user(payload.query())
                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .call()
                .content();

        return ResponseEntity.ok(new SearchResponse(payload.query(), answer));
    }
}


```

---

### 5. DTOs
> **File:** `src/main/java/com/example/ragdemo/dto/SearchRequestPayload.java`
```java
package com.example.simpleragapp.dto;

public record SearchRequestPayload(
        String query
) {
}

```

> **File:** `src/main/java/com/example/ragdemo/dto/SearchResponse.java`
```java
package com.example.simpleragapp.dto;

public record SearchResponse(
        String query,
        String answer
) {

}
```

### 6. Main Application Class

> **File:** `src/main/java/com/example/simpleragapp/SpringAiRagApplication.java`

```java

package com.example.simpleragapp;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SimpleRagAppApplication {

    public static void main(String[] args) {
        SpringApplication.run(SimpleRagAppApplication.class, args);
    }
}

```

---

## Running & Testing

### Step 1: Start the Spring Boot App

```bash
mvn spring-boot:run

```

### Step 2: Upload Knowledge Base Document

```bash
curl -X POST http://localhost:8080/api/v1/rag/upload \
  -F "file=@sample_document.txt"

```

### Step 3: Ask a Question

```bash
curl -X GET "http://localhost:8080/api/v1/rag/search?query=Your%20question%20here"

```