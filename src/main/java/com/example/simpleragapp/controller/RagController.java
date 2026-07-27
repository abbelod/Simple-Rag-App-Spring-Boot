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
//                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .call()
                .content();

        return ResponseEntity.ok(new SearchResponse(payload.query(), answer));
    }


}
