import dev.langchain4j.community.store.embedding.yugabytedb.YugabyteDBEmbeddingStore;
import dev.langchain4j.community.store.embedding.yugabytedb.YugabyteDBEngine;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.containers.wait.strategy.Wait;

import java.time.Duration;
import java.util.List;

public class YugabyteDBEmbeddingStoreExample {

    public static void main(String[] args) {

        DockerImageName dockerImageName = DockerImageName.parse("yugabytedb/yugabyte:latest");

        try (GenericContainer<?> yugabyteContainer = new GenericContainer<>(dockerImageName)
                .withExposedPorts(5433)
                .withCommand("bin/yugabyted", "start", "--daemon=false")
                .waitingFor(Wait.forListeningPort().withStartupTimeout(Duration.ofSeconds(30)))
                .withReuse(true)) {  // ✅ reuse container

            yugabyteContainer.start();

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            YugabyteDBEngine engine = YugabyteDBEngine.builder()
                    .host(yugabyteContainer.getHost())
                    .port(yugabyteContainer.getMappedPort(5433))
                    .database("yugabyte")
                    .username("yugabyte")
                    .password("yugabyte")
                    .usePostgreSQLDriver(true)
                    .build();

            EmbeddingStore<TextSegment> embeddingStore = YugabyteDBEmbeddingStore.builder()
                    .engine(engine)
                    .tableName("test_embeddings")
                    .dimension(embeddingModel.dimension())
                    .createTableIfNotExists(true)
                    .build();

            // ✅ Add data
            TextSegment segment1 = TextSegment.from("I like football.");
            embeddingStore.add(embeddingModel.embed(segment1).content(), segment1);

            TextSegment segment2 = TextSegment.from("The weather is good today.");
            embeddingStore.add(embeddingModel.embed(segment2).content(), segment2);

            // ✅ Query data
            Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();

            List<EmbeddingMatch<TextSegment>> relevant = embeddingStore
                    .search(EmbeddingSearchRequest.builder()
                            .queryEmbedding(queryEmbedding)
                            .maxResults(1)
                            .build())
                    .matches();

            EmbeddingMatch<TextSegment> match = relevant.get(0);

            System.out.println("\nSimilarity Score: " + match.score());
            System.out.println("Matched Text: " + match.embedded().text());
            System.out.println("\n✅ Example completed successfully!");

            engine.close();
            yugabyteContainer.stop();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
