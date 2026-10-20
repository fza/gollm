// Package llm provides a unified interface for interacting with various Language Learning Model providers.
// It abstracts away provider-specific implementations and provides a consistent API for text generation,
// prompt management, and error handling.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// LLM interface defines the methods that our internal language model should implement.
// It provides a unified way to interact with different LLM providers while abstracting
// away provider-specific details.
type LLM interface {
	// Generate produces text based on the given prompt and options.
	// Returns ErrorTypeRequest for request preparation failures,
	// ErrorTypeAPI for provider API errors, or ErrorTypeResponse for response processing issues.
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (response string, err error)

	// GenerateWithUsage behaves like Generate but returns the full Response,
	// including token usage and the prompt-cache read/creation breakdown when
	// the provider reports them. Use it when callers need per-call metrics
	// rather than just the generated text.
	GenerateWithUsage(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (*Response, error)

	// GenerateWithSchema generates text that conforms to a specific JSON schema.
	// Returns ErrorTypeInvalidInput for schema validation failures,
	// or other error types as per Generate.
	GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error)

	// Stream initiates a streaming response from the LLM.
	// Returns ErrorTypeUnsupported if the provider doesn't support streaming.
	Stream(ctx context.Context, prompt *Prompt, opts ...StreamOption) (TokenStream, error)

	// SupportsStreaming checks if the provider supports streaming responses.
	SupportsStreaming() bool

	// SetOption configures a provider-specific option.
	// Returns ErrorTypeInvalidInput if the option is not supported.
	SetOption(key string, value interface{})

	// SetLogLevel adjusts the logging verbosity.
	SetLogLevel(level utils.LogLevel)

	// SetEndpoint updates the API endpoint (primarily for local models).
	// Returns ErrorTypeProvider if the provider doesn't support endpoint configuration.
	SetEndpoint(endpoint string)

	// NewPrompt creates a new prompt instance.
	NewPrompt(input string) *Prompt

	// GetLogger returns the current logger instance.
	GetLogger() utils.Logger

	// SupportsJSONSchema checks if the provider supports JSON schema validation.
	SupportsJSONSchema() bool
}

// LLMImpl implements the LLM interface and manages interactions with specific providers.
// It handles provider communication, error management, and logging.
type LLMImpl struct {
	Provider      providers.Provider     // The underlying LLM provider
	Options       map[string]interface{} // Provider-specific options
	optionsMutex  sync.RWMutex           // Mutex to protect concurrent access to Options map
	client        *http.Client           // HTTP client for API requests
	logger        utils.Logger           // Logger for debugging and monitoring
	config        *config.Config         // Configuration settings
	MaxRetries    int                    // Maximum number of retry attempts
	RetryDelay    time.Duration          // Base delay for exponential retry backoff
	MaxRetryDelay time.Duration          // Cap on backoff growth (0 = uncapped)
}

// GenerateOption is a function type for configuring generation behavior.
type GenerateOption func(*GenerateConfig)

// GenerateConfig holds configuration options for text generation.
type GenerateConfig struct {
	UseJSONSchema bool // Whether to use JSON schema validation
}

// NewLLM creates a new LLM instance with the specified configuration.
// It initializes the appropriate provider and sets up logging and HTTP clients.
//
// Returns:
//   - Configured LLM instance
//   - ErrorTypeProvider if provider initialization fails
//   - ErrorTypeAuthentication if API key validation fails
func NewLLM(cfg *config.Config, logger utils.Logger, registry *providers.ProviderRegistry) (LLM, error) {
	extraHeaders := make(map[string]string)
	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	// Check if API key is empty (skip for local providers that don't require auth)
	apiKey := cfg.APIKeys[cfg.Provider]
	isLocalProvider := cfg.Provider == "ollama" || cfg.Provider == "lmstudio" || cfg.Provider == "vllm"
	if apiKey == "" && !isLocalProvider {
		return nil, NewLLMError(ErrorTypeAuthentication, "empty API key", nil)
	}

	provider, err := registry.Get(cfg.Provider, apiKey, cfg.Model, extraHeaders)

	if err != nil {
		return nil, err
	}

	provider.SetDefaultOptions(cfg)

	llmClient := &LLMImpl{
		Provider:      provider,
		client:        &http.Client{Timeout: cfg.Timeout},
		logger:        logger,
		config:        cfg,
		MaxRetries:    cfg.MaxRetries,
		RetryDelay:    cfg.RetryDelay,
		MaxRetryDelay: cfg.MaxRetryDelay,
		Options:       make(map[string]interface{}),
	}

	return llmClient, nil
}

// SetOption sets a provider-specific option with the given key and value.
// The option is logged at debug level for troubleshooting.
func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.optionsMutex.Lock()
	defer l.optionsMutex.Unlock()

	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

// SetEndpoint updates the API endpoint for the provider.
// This is primarily used for local models like Ollama.
func (l *LLMImpl) SetEndpoint(endpoint string) {
	// This is a no-op for non-Ollama providers
	l.logger.Debug("SetEndpoint called on non-Ollama provider", "endpoint", endpoint)
}

// SetLogLevel updates the logging verbosity level.
func (l *LLMImpl) SetLogLevel(level utils.LogLevel) {
	l.logger.Debug("Setting internal LLM log level", "new_level", level)
	l.logger.SetLevel(level)
}

// GetLogger returns the current logger instance.
func (l *LLMImpl) GetLogger() utils.Logger {
	return l.logger
}

// NewPrompt creates a new prompt instance with the given input text.
func (l *LLMImpl) NewPrompt(prompt string) *Prompt {
	return &Prompt{Input: prompt}
}

// SupportsJSONSchema checks if the current provider supports JSON schema validation.
func (l *LLMImpl) SupportsJSONSchema() bool {
	return l.Provider.SupportsJSONSchema()
}

// Generate produces text based on the given prompt and options.
// It handles retries, logging, and error management.
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	resp, err := l.generate(ctx, prompt, opts...)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// GenerateWithUsage behaves like Generate but returns the full Response,
// including token usage and the prompt-cache read/creation breakdown when the
// provider reports them.
func (l *LLMImpl) GenerateWithUsage(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (*Response, error) {
	return l.generate(ctx, prompt, opts...)
}

// generate runs the retry loop around a single attemptGenerate and returns the
// full Response. Generate and GenerateWithUsage are thin wrappers over it.
func (l *LLMImpl) generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (*Response, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}
	// Set the system prompt in the LLM's options
	if prompt.SystemPrompt != "" {
		l.SetOption("system_prompt", prompt.SystemPrompt)
	}
	var lastErr error
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text", "provider", l.Provider.Name(), "prompt", prompt.String(), "system_prompt", prompt.SystemPrompt, "attempt", attempt+1)
		// Pass the entire Prompt struct to attemptGenerate
		resp, err := l.attemptGenerate(ctx, prompt)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		proceed, ctxErr := l.shouldRetry(ctx, attempt, err)
		if ctxErr != nil {
			return nil, ctxErr
		}
		if !proceed {
			break
		}
	}
	// Surface the underlying error rather than masking it with an attempt
	// count — callers classify on the typed error to decide what to do next.
	return nil, lastErr
}

// attemptGenerate makes a single attempt to generate text using the provider.
// It handles request preparation, API communication, and response processing.
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt *Prompt) (*Response, error) {
	// Create a new options map that includes both l.Options and prompt-specific options
	options := make(map[string]interface{})

	// Safely read from the Options map
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	// Add Tools and ToolChoice to options
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if len(prompt.ToolChoice) > 0 {
		options["tool_choice"] = prompt.ToolChoice
	}

	// Add Images to options for vision-capable models
	if prompt.HasImages() {
		options["images"] = prompt.Images
	}

	var reqBody []byte
	var err error

	// Check if we have structured messages
	l.optionsMutex.RLock()
	structuredMessages, hasStructuredMessages := l.Options["structured_messages"]
	l.optionsMutex.RUnlock()

	// Check if we have structured messages
	if hasStructuredMessages {
		// Use the structured messages API if the provider supports it
		if prepareWithMessages, ok := l.Provider.(interface {
			PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)
		}); ok {
			// Convert to the expected type
			messages, ok := structuredMessages.([]types.MemoryMessage)
			if ok {
				l.logger.Debug("Using structured messages API", "message_count", len(messages))
				reqBody, err = prepareWithMessages.PrepareRequestWithMessages(messages, options)
			} else {
				l.logger.Warn("Invalid structured_messages format", "type", fmt.Sprintf("%T", structuredMessages))
				// Fall back to regular prepare
				reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
			}
		} else {
			l.logger.Debug("Provider does not support structured messages API", "provider", l.Provider.Name())
			// Provider doesn't support structured messages, fall back to normal request
			reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
		}
	} else {
		// Standard request preparation
		reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
	}

	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Full request body", "body", string(reqBody))
	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	l.logger.Debug("Full API request", "method", req.Method, "url", req.URL.String(), "headers", req.Header, "body", string(reqBody))
	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
		l.logger.Debug("Request header", "provider", l.Provider.Name(), "key", k, "value", v)
	}
	resp, err := l.client.Do(req)
	if err != nil {
		return nil, classifyNetworkError(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	// Log the full API response
	l.logger.Debug("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		apiErr := classifyAPIError(resp.StatusCode, resp.Header, body)
		if apiErr.Retryable {
			// Transient — log quietly so a batch of retries does not flood.
			l.logger.Debug("Retryable API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "retry_after", apiErr.RetryAfter.String())
		} else {
			// Permanent — the returned error carries the provider's code and
			// message, so the caller decides whether it is worth reporting; a
			// caller that recovers from it must not leave an error in the log.
			l.logger.Debug("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		}
		return nil, apiErr
	}

	// Extract and log caching information
	var fullResponse map[string]interface{}
	if err := json.Unmarshal(body, &fullResponse); err != nil {
		// Try parsing as JSONL if JSON parsing fails
		lines := bytes.Split(bytes.TrimSpace(body), []byte("\n"))
		if len(lines) > 0 {
			// the last line is *usually* the full response
			if err := json.Unmarshal(lines[len(lines)-1], &fullResponse); err != nil {
				l.logger.Warn("Failed to parse response as both JSON and JSONL", "error", err)
			}
		}
	}

	// Process usage information regardless of format
	if usage, ok := fullResponse["usage"].(map[string]interface{}); ok {
		l.logger.Debug("Usage information", "usage", usage)
		cacheInfo := map[string]interface{}{
			"cache_creation_input_tokens": usage["cache_creation_input_tokens"],
			"cache_read_input_tokens":     usage["cache_read_input_tokens"],
		}
		l.logger.Debug("Cache information", "info", cacheInfo)
	} else {
		l.logger.Debug("Cache information not available in the response")
	}

	usage := usageFromResponseMap(fullResponse)
	finishReason := finishReasonFromResponseMap(fullResponse)
	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		if finishReason != "" {
			err = &NoAnswerError{FinishReason: finishReason, Usage: usage, Err: err}
		}
		return nil, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}
	l.logger.Debug("Text generated successfully", "result", result)
	return &Response{Content: result, Usage: usage, FinishReason: finishReason}, nil
}

// finishReasonFromResponseMap reads why the model stopped from a decoded
// provider response: OpenAI's choices[0].finish_reason, Anthropic's
// stop_reason or Ollama's done_reason.
func finishReasonFromResponseMap(full map[string]interface{}) string {
	if choices, ok := full["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if reason, ok := choice["finish_reason"].(string); ok {
				return reason
			}
		}
	}
	for _, key := range []string{"stop_reason", "done_reason"} {
		if reason, ok := full[key].(string); ok {
			return reason
		}
	}
	return ""
}

// usageFromResponseMap extracts token usage from a decoded provider response.
// It reads the OpenAI-style (prompt/completion/total) and Anthropic-style
// (input/output plus cache read/creation) names from a usage object, and
// Ollama's counts, which sit at the top level of the final response object
// instead. Returns nil when the response reports no usage at all — an absence
// stays an absence rather than becoming a zero that reads as a measurement.
func usageFromResponseMap(full map[string]interface{}) *Usage {
	geti := func(m map[string]interface{}, k string) int {
		if v, ok := m[k].(float64); ok {
			return int(v)
		}
		return 0
	}

	if raw, ok := full["usage"].(map[string]interface{}); ok {
		u := &Usage{
			PromptTokens:             geti(raw, "prompt_tokens"),
			CompletionTokens:         geti(raw, "completion_tokens"),
			TotalTokens:              geti(raw, "total_tokens"),
			InputTokens:              geti(raw, "input_tokens"),
			OutputTokens:             geti(raw, "output_tokens"),
			CacheCreationInputTokens: geti(raw, "cache_creation_input_tokens"),
			CacheReadInputTokens:     geti(raw, "cache_read_input_tokens"),
		}
		// OpenAI reports automatic prompt-cache hits nested under
		// prompt_tokens_details.cached_tokens instead of a top-level field.
		if u.CacheReadInputTokens == 0 {
			if details, ok := raw["prompt_tokens_details"].(map[string]interface{}); ok {
				u.CacheReadInputTokens = geti(details, "cached_tokens")
			}
		}
		// Chat Completions nests reasoning under completion_tokens_details,
		// the Responses API under output_tokens_details.
		for _, key := range []string{"completion_tokens_details", "output_tokens_details"} {
			if details, ok := raw[key].(map[string]interface{}); ok && u.ReasoningTokens == 0 {
				u.ReasoningTokens = geti(details, "reasoning_tokens")
			}
		}
		return u
	}

	// Ollama names its counts prompt_eval_count and eval_count and puts them
	// beside the generated text, on the object carrying done: true.
	prompt, completion := geti(full, "prompt_eval_count"), geti(full, "eval_count")
	if prompt == 0 && completion == 0 {
		return nil
	}
	return &Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
	}
}

// GenerateWithSchema generates text that conforms to a specific JSON schema.
// It handles retries, logging, and error management.
//
// Returns:
//   - Generated text response
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per Generate
func (l *LLMImpl) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	var result string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with schema", "provider", l.Provider.Name(), "prompt", prompt.String(), "attempt", attempt+1)

		result, _, lastErr = l.attemptGenerateWithSchema(ctx, prompt.String(), schema)
		if lastErr == nil {
			return result, nil
		}

		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
}

// attemptGenerateWithSchema makes a single attempt to generate text using the provider and a JSON schema.
// It handles request preparation, API communication, and response processing.
//
// Returns:
//   - Generated text response
//   - Full prompt used for generation
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per attemptGenerate
func (l *LLMImpl) attemptGenerateWithSchema(ctx context.Context, prompt string, schema interface{}) (string, string, error) {
	var reqBody []byte
	var err error
	var fullPrompt string

	l.optionsMutex.RLock()
	options := make(map[string]interface{})
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	if l.SupportsJSONSchema() {
		reqBody, err = l.Provider.PrepareRequestWithSchema(prompt, options, schema)
		fullPrompt = prompt
	} else {
		fullPrompt = l.preparePromptWithSchema(prompt, schema)
		reqBody, err = l.Provider.PrepareRequest(fullPrompt, options)
	}

	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	if resp.StatusCode != http.StatusOK {
		l.logger.Error("API error", "provider", l.Provider.Name(), "status", resp.StatusCode, "body", string(body))
		return "", fullPrompt, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	result, err := l.Provider.ParseResponse(body)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	// Validate the result against the schema
	if err := ValidateAgainstSchema(result, schema); err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
	}

	l.logger.Debug("Text generated successfully", "result", result)
	return result, fullPrompt, nil
}

// preparePromptWithSchema prepares a prompt with a JSON schema for providers that do not support JSON schema validation.
// Returns the original prompt if schema marshaling fails (with a warning log).
func (l *LLMImpl) preparePromptWithSchema(prompt string, schema interface{}) string {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		l.logger.Warn("Failed to marshal schema", "error", err)
		return prompt
	}

	return fmt.Sprintf("%s\n\nPlease provide your response in JSON format according to this schema:\n%s", prompt, string(schemaJSON))
}

// Stream initiates a streaming response from the LLM.
func (l *LLMImpl) Stream(ctx context.Context, prompt *Prompt, opts ...StreamOption) (TokenStream, error) {
	if !l.SupportsStreaming() {
		return nil, NewLLMError(ErrorTypeUnsupported, "streaming not supported by provider", nil)
	}

	// Apply stream options
	config := &StreamConfig{
		BufferSize: 100,
		RetryStrategy: &DefaultRetryStrategy{
			MaxRetries:  l.MaxRetries,
			InitialWait: l.RetryDelay,
			MaxWait:     l.RetryDelay * 10,
		},
	}
	for _, opt := range opts {
		opt(config)
	}

	// Prepare request with streaming enabled
	options := make(map[string]interface{})
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()
	options["stream"] = true

	body, err := l.Provider.PrepareStreamRequest(prompt.String(), options)
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to prepare stream request", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to create stream request", err)
	}

	// Add headers
	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	// Make request
	resp, err := l.client.Do(req)
	if err != nil {
		return nil, NewLLMError(ErrorTypeAPI, "failed to make stream request", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, NewLLMError(ErrorTypeAPI, fmt.Sprintf("API error: status code %d", resp.StatusCode), nil)
	}

	// Create and return stream
	return newProviderStream(resp.Body, l.Provider, config), nil
}

// SupportsStreaming checks if the provider supports streaming responses.
func (l *LLMImpl) SupportsStreaming() bool {
	return l.Provider.SupportsStreaming()
}

// providerStream implements TokenStream for a specific provider
type providerStream struct {
	decoder       StreamDecoder
	provider      providers.Provider
	config        *StreamConfig
	buffer        []byte
	currentIndex  int
	retryStrategy RetryStrategy
}

func newProviderStream(reader io.ReadCloser, provider providers.Provider, config *StreamConfig) *providerStream {
	var decoder StreamDecoder
	if provider.Name() == "ollama" {
		decoder = NewNDJSONDecoder(reader)
	} else {
		decoder = NewSSEDecoder(reader)
	}

	return &providerStream{
		decoder:       decoder,
		provider:      provider,
		config:        config,
		buffer:        make([]byte, 0, 4096),
		currentIndex:  0,
		retryStrategy: config.RetryStrategy,
	}
}

func (s *providerStream) Next(ctx context.Context) (*StreamToken, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			if !s.decoder.Next() {
				if err := s.decoder.Err(); err != nil {
					if s.retryStrategy.ShouldRetry(err) {
						time.Sleep(s.retryStrategy.NextDelay())
						continue
					}
					return nil, err
				}
				return nil, io.EOF
			}

			event := s.decoder.Event()
			if len(event.Data) == 0 {
				continue
			}

			// Process the event
			token, err := s.provider.ParseStreamResponse(event.Data)
			if err != nil {
				if err.Error() == "skip token" {
					continue
				}
				if err == io.EOF {
					return nil, io.EOF
				}
				continue // Not enough data or malformed
			}

			// Create and return token
			return &StreamToken{
				Text:  token,
				Type:  event.Type,
				Index: s.currentIndex,
			}, nil
		}
	}
}

func (s *providerStream) Close() error {
	return nil
}
