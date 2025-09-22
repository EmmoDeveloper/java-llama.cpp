# Server Benchmarking

The `ServerBenchmark` tool provides comprehensive performance testing for llama.cpp servers, measuring throughput, latency, and token generation speed.

## Features

- **Throughput measurement** - Requests per second
- **Latency analysis** - P50, P90, P99 percentiles
- **Token generation speed** - Tokens per second
- **Concurrent request handling** - Multiple client simulation
- **Various workload patterns** - Random, file-based, or predefined prompts
- **Detailed metrics** - Per-request timing and token counts
- **Result export** - JSON format for analysis

## Usage

### Basic Usage

```java
ServerBenchmark.BenchmarkConfig config = new ServerBenchmark.BenchmarkConfig()
    .serverUrl("http://localhost:8080")
    .requests(100)
    .concurrency(5)
    .maxTokens(50);

ServerBenchmark benchmark = new ServerBenchmark(config);
ServerBenchmark.BenchmarkResults results = benchmark.run();
System.out.println(results.toFormattedString());
```

### Command Line Interface

```bash
# Basic benchmark
java de.kherud.llama.benchmark.ServerBenchmark \
    --server http://localhost:8080 \
    --requests 1000 \
    --concurrency 10

# Advanced configuration
java de.kherud.llama.benchmark.ServerBenchmark \
    --server http://localhost:8080 \
    --requests 500 \
    --concurrency 20 \
    --dataset shakespeare \
    --max-tokens 100 \
    --temperature 0.8 \
    --output results.json \
    --verbose
```

## Configuration Options

### Server Settings
```java
config.serverUrl("http://localhost:8080");  // Server URL
config.timeout(Duration.ofMinutes(5));      // Request timeout
```

### Workload Settings
```java
config.requests(1000);                      // Total number of requests
config.concurrency(10);                     // Concurrent clients
config.maxTokens(100);                      // Max tokens to generate
```

### Dataset Options
```java
// Random prompts
config.dataset("random")
    .promptLength(50, 200);                 // Word count range

// File-based prompts (one per line)
config.dataset("file:/path/to/prompts.txt");

// Predefined dataset
config.dataset("shakespeare");
```

### Generation Parameters
```java
config.addGenerationParam("temperature", 0.7);
config.addGenerationParam("top_p", 0.9);
config.addGenerationParam("top_k", 40);
config.addGenerationParam("repeat_penalty", 1.1);
```

### Output Settings
```java
config.verbose(true);                       // Enable detailed logging
config.outputFile("benchmark_results.json"); // Save results to file
```

## Metrics Collected

### Request-Level Metrics
- **Request ID** - Unique identifier
- **Duration** - Total request time (ms)
- **Prompt tokens** - Input token count
- **Completion tokens** - Generated token count
- **Tokens per second** - Generation speed
- **Success/failure** - Request status
- **Error message** - If failed

### Aggregate Results
- **Total requests** - All submitted requests
- **Success rate** - Percentage successful
- **Requests per second** - Overall throughput
- **Latency percentiles** - P50, P90, P99 response times
- **Token statistics** - Total and average generation speed

## Sample Output

```
========== BENCHMARK RESULTS ==========
Total Requests:        1000
Successful:            995 (99.5%)
Failed:                5
Total Duration:        45.67 seconds
Requests/Second:       21.90

--- Latency (ms) ---
Average:               456.23
Min:                   89.45
Max:                   2347.12
P50:                   423.67
P90:                   745.23
P99:                   1234.56

--- Tokens ---
Total Prompt:          52341
Total Completion:      87654
Avg Tokens/Second:     15.67
Total Tokens/Second:   1920.45
========================================
```

## Performance Analysis

### Throughput Optimization
- **Concurrent clients** - More clients can improve utilization but may increase latency
- **Batch size** - Server-side batching improves throughput
- **Model size** - Smaller models generally have higher throughput

### Latency Analysis
- **P50** - Typical user experience
- **P90** - Good user experience threshold
- **P99** - Worst-case scenarios

### Token Generation Speed
- **Per-request speed** - Individual request performance
- **Aggregate speed** - Overall system throughput
- **Hardware utilization** - GPU/CPU efficiency

## Workload Patterns

### Load Testing
```java
// Sustained load test
config.requests(10000)
    .concurrency(50)
    .dataset("random");
```

### Stress Testing
```java
// Peak capacity test
config.requests(1000)
    .concurrency(100)
    .timeout(Duration.ofSeconds(30));
```

### Endurance Testing
```java
// Long-running stability test
config.requests(50000)
    .concurrency(20)
    .dataset("file:large_dataset.txt");
```

## Integration with CI/CD

### Performance Regression Detection
```java
ServerBenchmark.BenchmarkResults results = benchmark.run();
if (results.requestsPerSecond < MINIMUM_THROUGHPUT) {
    throw new AssertionError("Performance regression detected");
}
if (results.p99LatencyMs > MAXIMUM_LATENCY) {
    throw new AssertionError("Latency regression detected");
}
```

### Automated Reporting
```java
// Save detailed results
results.saveToFile("benchmark_" + timestamp + ".json");

// Log summary
LOGGER.info("Benchmark completed: " +
    results.requestsPerSecond + " req/s, " +
    results.p50LatencyMs + "ms P50");
```

## Server Health Checks

The benchmark automatically performs health checks before starting:

```java
// Automatic health check
GET /health

// Expected response: 200 OK
```

If the server is not healthy, the benchmark will fail early with a clear error message.

## Error Handling

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| Connection refused | Server not running | Start llama.cpp server |
| Timeout | Server overloaded | Reduce concurrency or increase timeout |
| 404 Not Found | Wrong endpoint | Verify server URL and API path |
| 500 Internal Error | Server error | Check server logs |

### Retry Logic
The benchmark includes automatic retry for transient errors:
- Network timeouts
- Connection resets
- Temporary server errors

### Graceful Degradation
- Failed requests are recorded but don't stop the benchmark
- Partial results are still meaningful
- Error rates are included in final metrics

## Advanced Features

### Custom Request Bodies
```java
Map<String, Object> customParams = new HashMap<>();
customParams.put("stop", Arrays.asList("\n", "Human:"));
customParams.put("logit_bias", Map.of("13", -100)); // Avoid token 13
config.addGenerationParam("stop", customParams.get("stop"));
```

### Response Validation
```java
// Extend ServerBenchmark to add custom validation
public class CustomBenchmark extends ServerBenchmark {
    @Override
    protected void validateResponse(JsonNode response) {
        // Custom validation logic
        if (!response.has("content")) {
            throw new ValidationException("Missing content field");
        }
    }
}
```

### Multi-Server Testing
```java
// Test multiple server instances
List<String> servers = Arrays.asList(
    "http://server1:8080",
    "http://server2:8080",
    "http://server3:8080"
);

for (String server : servers) {
    ServerBenchmark benchmark = new ServerBenchmark(
        config.serverUrl(server)
    );
    results.put(server, benchmark.run());
}
```

## Best Practices

1. **Warm up the server** - Run a small benchmark first to load the model
2. **Use realistic workloads** - Match your actual usage patterns
3. **Monitor server resources** - CPU, memory, GPU utilization
4. **Test different scenarios** - Varying prompt lengths, generation parameters
5. **Baseline measurements** - Establish performance baselines for comparison
6. **Regular testing** - Include benchmarks in your deployment pipeline