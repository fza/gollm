package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/teilomillet/gollm/config"
)

// TestAnthropicEndpoint verifies that the provider targets Anthropic by
// default and a Messages API service when a base URL is configured, whichever
// of the accepted URL forms names it.
func TestAnthropicEndpoint(t *testing.T) {
	testCases := []struct {
		name             string
		endpoint         string
		expectedEndpoint string
	}{
		{name: "unset endpoint targets Anthropic", endpoint: "", expectedEndpoint: "https://api.anthropic.com/v1/messages"},
		{name: "base URL", endpoint: "https://gateway.example/compat", expectedEndpoint: "https://gateway.example/compat/v1/messages"},
		{name: "base URL already has /v1", endpoint: "https://gateway.example/compat/v1", expectedEndpoint: "https://gateway.example/compat/v1/messages"},
		{name: "trailing slash", endpoint: "https://gateway.example/compat/", expectedEndpoint: "https://gateway.example/compat/v1/messages"},
		{name: "full path already provided", endpoint: "https://gateway.example/compat/v1/messages", expectedEndpoint: "https://gateway.example/compat/v1/messages"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			provider := NewAnthropicProvider("fake-api-key", "claude-sonnet-5", nil).(*AnthropicProvider)
			provider.SetEndpoint(tc.endpoint)

			assert.Equal(t, tc.expectedEndpoint, provider.Endpoint())
		})
	}
}

// TestAnthropicEndpointFromConfig verifies that the configured base URL
// reaches the provider through the default options, the path NewLLM composes.
func TestAnthropicEndpointFromConfig(t *testing.T) {
	provider := NewAnthropicProvider("fake-api-key", "claude-sonnet-5", nil).(*AnthropicProvider)
	provider.SetDefaultOptions(&config.Config{AnthropicEndpoint: "https://gateway.example/compat"})

	assert.Equal(t, "https://gateway.example/compat/v1/messages", provider.Endpoint())
}

// TestAnthropicEffortNestsUnderOutputConfig verifies that the flat "effort"
// option is sent as output_config.effort — the shape the Anthropic API
// expects — and never as an unknown top-level parameter (HTTP 400).
func TestAnthropicEffortNestsUnderOutputConfig(t *testing.T) {
	provider, ok := NewAnthropicProvider("fake-api-key", "claude-sonnet-5", nil).(*AnthropicProvider)
	assert.True(t, ok, "Provider should be of type *AnthropicProvider")

	body, err := provider.PrepareRequest("hello", map[string]interface{}{"effort": "low", "max_tokens": 4096})
	assert.NoError(t, err)

	var req map[string]interface{}
	assert.NoError(t, json.Unmarshal(body, &req))

	_, hasFlatEffort := req["effort"]
	assert.False(t, hasFlatEffort, "effort must not be sent top-level (Anthropic rejects unknown parameters)")

	outputConfig, ok := req["output_config"].(map[string]interface{})
	assert.True(t, ok, "output_config must be present")
	assert.Equal(t, "low", outputConfig["effort"], "effort level must be preserved under output_config")
}
