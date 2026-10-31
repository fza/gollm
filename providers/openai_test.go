package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/teilomillet/gollm/config"
)

// TestTokenBudgetSentAsMaxCompletionTokens verifies that every request path
// carries the token budget as max_completion_tokens and never max_tokens,
// whatever the model is called: reasoning models reject max_tokens with HTTP
// 400, and a gateway's model names (openai/gpt-5-mini, anthropic/...) defeat
// any prefix table.
func TestTokenBudgetSentAsMaxCompletionTokens(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)
	for _, model := range []string{"gpt-4", "gpt-4o-mini", "o1-preview", "gpt-5.4-2026-03-05", "openai/gpt-5-mini", "anthropic/claude-sonnet-4-6"} {
		for name, prepare := range map[string]func(p *OpenAIProvider, opts map[string]interface{}) ([]byte, error){
			"PrepareRequest": func(p *OpenAIProvider, opts map[string]interface{}) ([]byte, error) {
				return p.PrepareRequest("hello", opts)
			},
			"PrepareRequestWithSchema": func(p *OpenAIProvider, opts map[string]interface{}) ([]byte, error) {
				return p.PrepareRequestWithSchema("hello", opts, schema)
			},
			"PrepareStreamRequest": func(p *OpenAIProvider, opts map[string]interface{}) ([]byte, error) {
				return p.PrepareStreamRequest("hello", opts)
			},
		} {
			t.Run(model+"/"+name, func(t *testing.T) {
				provider := NewOpenAIProvider("fake-api-key", model, nil).(*OpenAIProvider)
				provider.SetDefaultOptions(&config.Config{MaxTokens: 4096})

				body, err := prepare(provider, map[string]interface{}{"max_tokens": 2048})
				assert.NoError(t, err)

				var req map[string]interface{}
				assert.NoError(t, json.Unmarshal(body, &req))
				_, hasMaxTokens := req["max_tokens"]
				assert.False(t, hasMaxTokens, "max_tokens must not be sent")
				assert.Equal(t, float64(2048), req["max_completion_tokens"], "the per-call budget must survive as max_completion_tokens")
			})
		}
	}
}

// TestTemperatureSentOnlyWhenConfigured verifies that the config's default
// temperature stays out of the request — reasoning models reject any value
// but their own default — while a configured one is sent to every model.
func TestTemperatureSentOnlyWhenConfigured(t *testing.T) {
	for _, model := range []string{"gpt-4o", "gpt-5.6-luna", "openai/gpt-5-mini"} {
		t.Run(model, func(t *testing.T) {
			defaults := NewOpenAIProvider("fake-api-key", model, nil).(*OpenAIProvider)
			defaults.SetDefaultOptions(&config.Config{Temperature: 0.7, MaxTokens: 4096})
			configured := NewOpenAIProvider("fake-api-key", model, nil).(*OpenAIProvider)
			configured.SetDefaultOptions(&config.Config{Temperature: 0.2, TemperatureSet: true, MaxTokens: 4096})

			for provider, want := range map[*OpenAIProvider]interface{}{defaults: nil, configured: 0.2} {
				body, err := provider.PrepareRequest("hello", nil)
				assert.NoError(t, err)
				var req map[string]interface{}
				assert.NoError(t, json.Unmarshal(body, &req))
				assert.Equal(t, want, req["temperature"])
			}
		})
	}
}

// TestOpenAIEndpoint verifies that the provider targets OpenAI by default and
// an OpenAI-compatible service when a base URL is configured, whichever of the
// accepted URL forms names it.
func TestOpenAIEndpoint(t *testing.T) {
	testCases := []struct {
		name             string
		endpoint         string
		expectedEndpoint string
	}{
		{
			name:             "unset endpoint targets OpenAI",
			endpoint:         "",
			expectedEndpoint: "https://api.openai.com/v1/chat/completions",
		},
		{
			name:             "host only",
			endpoint:         "https://gateway.example/openai",
			expectedEndpoint: "https://gateway.example/openai/v1/chat/completions",
		},
		{
			name:             "base URL already has /v1",
			endpoint:         "https://gateway.example/openai/v1",
			expectedEndpoint: "https://gateway.example/openai/v1/chat/completions",
		},
		{
			name:             "base URL with /v1 and trailing slash",
			endpoint:         "https://gateway.example/openai/v1/",
			expectedEndpoint: "https://gateway.example/openai/v1/chat/completions",
		},
		{
			name:             "full path already provided",
			endpoint:         "https://gateway.example/openai/v1/chat/completions",
			expectedEndpoint: "https://gateway.example/openai/v1/chat/completions",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewOpenAIProvider("fake-api-key", "gpt-4o-mini", nil).(*OpenAIProvider)
			provider.SetEndpoint(tc.endpoint)

			assert.Equal(t, tc.expectedEndpoint, provider.Endpoint())
		})
	}
}

// TestOpenAIEndpointFromConfig verifies that the configured base URL reaches
// the provider through the default options, the path NewLLM composes.
func TestOpenAIEndpointFromConfig(t *testing.T) {
	provider := NewOpenAIProvider("fake-api-key", "gpt-4o-mini", nil).(*OpenAIProvider)
	provider.SetDefaultOptions(&config.Config{OpenAIEndpoint: "https://gateway.example/openai"})

	assert.Equal(t, "https://gateway.example/openai/v1/chat/completions", provider.Endpoint())
}
