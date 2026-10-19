package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/teilomillet/gollm/config"
)

// TestNeedsMaxCompletionTokens verifies that the needsMaxCompletionTokens function
// correctly identifies models that require max_completion_tokens instead of max_tokens
func TestNeedsMaxCompletionTokens(t *testing.T) {
	testCases := []struct {
		modelName      string
		expectedResult bool
		description    string
	}{
		{"gpt-4", false, "Standard GPT-4 model should not use max_completion_tokens"},
		{"gpt-4-turbo", false, "GPT-4 Turbo model should not use max_completion_tokens"},
		{"gpt-3.5-turbo", false, "GPT-3.5 Turbo model should not use max_completion_tokens"},
		{"o1-preview", true, "o1-preview model should use max_completion_tokens"},
		{"o-preview", true, "o-preview model should use max_completion_tokens"},
		{"gpt-4o", true, "GPT-4o model should use max_completion_tokens"},
		{"gpt-4o-mini", true, "GPT-4o mini model should use max_completion_tokens"},
		{"gpt-5", true, "GPT-5 model should use max_completion_tokens"},
		{"gpt-5-mini", true, "GPT-5 mini model should use max_completion_tokens"},
		{"gpt-5.4-2026-03-05", true, "Dated GPT-5 model should use max_completion_tokens"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Create a provider with the test model
			provider := NewOpenAIProvider("fake-api-key", tc.modelName, nil)

			// Get the OpenAIProvider concrete type from the Provider interface
			openAIProvider, ok := provider.(*OpenAIProvider)
			assert.True(t, ok, "Provider should be of type *OpenAIProvider")

			// Test the needsMaxCompletionTokens function
			result := openAIProvider.needsMaxCompletionTokens()
			assert.Equal(t, tc.expectedResult, result, "needsMaxCompletionTokens returned unexpected result for model %s", tc.modelName)
		})
	}
}

// TestPrepareRequestGPT5UsesMaxCompletionTokens verifies that a gpt-5 request
// carries max_completion_tokens and never max_tokens — the parameter gpt-5-class
// models reject with HTTP 400.
func TestPrepareRequestGPT5UsesMaxCompletionTokens(t *testing.T) {
	provider, ok := NewOpenAIProvider("fake-api-key", "gpt-5.4-2026-03-05", nil).(*OpenAIProvider)
	assert.True(t, ok, "Provider should be of type *OpenAIProvider")

	body, err := provider.PrepareRequest("hello", map[string]interface{}{"max_tokens": 4096})
	assert.NoError(t, err)

	var req map[string]interface{}
	assert.NoError(t, json.Unmarshal(body, &req))

	_, hasMaxTokens := req["max_tokens"]
	maxCompletion, hasMaxCompletion := req["max_completion_tokens"]
	assert.False(t, hasMaxTokens, "gpt-5 request must not send max_tokens (OpenAI rejects it with HTTP 400)")
	assert.True(t, hasMaxCompletion, "gpt-5 request must send max_completion_tokens instead")
	assert.Equal(t, float64(4096), maxCompletion, "token budget must be preserved across the conversion")
}

// TestDefaultTemperatureSkippedForReasoningModels verifies that the configured
// default temperature is not sent to reasoning models (o-series, gpt-5.x) —
// they reject any non-default temperature with HTTP 400 — while gpt-4-class
// models keep receiving it.
func TestDefaultTemperatureSkippedForReasoningModels(t *testing.T) {
	testCases := []struct {
		modelName      string
		expectedResult bool
		description    string
	}{
		{"gpt-4", true, "GPT-4 accepts temperature"},
		{"gpt-4o", true, "GPT-4o accepts temperature"},
		{"gpt-3.5-turbo", true, "GPT-3.5 Turbo accepts temperature"},
		{"o1-preview", false, "o1 rejects non-default temperature"},
		{"gpt-5", false, "GPT-5 rejects non-default temperature"},
		{"gpt-5.6-luna", false, "GPT-5.6 Luna rejects non-default temperature"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			provider, ok := NewOpenAIProvider("fake-api-key", tc.modelName, nil).(*OpenAIProvider)
			assert.True(t, ok, "Provider should be of type *OpenAIProvider")

			provider.SetDefaultOptions(&config.Config{Temperature: 0.7, MaxTokens: 4096})

			body, err := provider.PrepareRequest("hello", nil)
			assert.NoError(t, err)

			var req map[string]interface{}
			assert.NoError(t, json.Unmarshal(body, &req))

			_, hasTemperature := req["temperature"]
			assert.Equal(t, tc.expectedResult, hasTemperature, "temperature presence unexpected for model %s", tc.modelName)
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

// TestSystemPromptRole verifies that the system prompt goes out as a developer
// message to OpenAI's own endpoint and as a system message to any other
// OpenAI-compatible endpoint, with the provider options passed through as set.
func TestSystemPromptRole(t *testing.T) {
	testCases := []struct {
		name         string
		endpoint     string
		expectedRole string
	}{
		{name: "unset endpoint", endpoint: "", expectedRole: "developer"},
		{name: "OpenAI base URL", endpoint: "https://api.openai.com/v1", expectedRole: "developer"},
		{name: "OpenAI full path", endpoint: "https://api.openai.com/v1/chat/completions", expectedRole: "developer"},
		{name: "OpenAI host in upper case", endpoint: "https://API.OpenAI.com", expectedRole: "developer"},
		{name: "melious", endpoint: "https://api.melious.ai/v1/chat/completions", expectedRole: "system"},
		{name: "gateway", endpoint: "https://gateway.example/openai", expectedRole: "system"},
		{name: "OpenAI host as a subdomain elsewhere", endpoint: "https://api.openai.com.gateway.example/v1", expectedRole: "system"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewOpenAIProvider("fake-api-key", "glm-5.3-flash", nil).(*OpenAIProvider)
			provider.SetEndpoint(tc.endpoint)
			provider.SetOption("reasoning_effort", "medium")

			body, err := provider.PrepareRequest("hello", map[string]interface{}{"system_prompt": "be terse"})
			assert.NoError(t, err)

			var req struct {
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
				ReasoningEffort string `json:"reasoning_effort"`
			}
			assert.NoError(t, json.Unmarshal(body, &req))

			if assert.Len(t, req.Messages, 2, "request should carry the system prompt and the user prompt") {
				assert.Equal(t, tc.expectedRole, req.Messages[0].Role, "system prompt role should follow the endpoint")
				assert.Equal(t, "be terse", req.Messages[0].Content, "system prompt should be sent unchanged")
			}
			assert.Equal(t, "medium", req.ReasoningEffort, "provider options should reach the request unchanged")
		})
	}
}
