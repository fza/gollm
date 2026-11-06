package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/teilomillet/gollm/config"
)

// TestNoOutputCapWhenMaxTokensUnset verifies that a zero MaxTokens sends no
// output cap on OpenAI, so a reasoning model keeps its own output limit, while
// a positive one is sent as the cap.
func TestNoOutputCapWhenMaxTokensUnset(t *testing.T) {
	testCases := []struct {
		description string
		model       string
	}{
		{"gateway model", "glm-5.3-flash"},
		{"reasoning model", "gpt-5"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			for _, maxTokens := range []int{0, 2048} {
				provider := NewOpenAIProvider("fake-api-key", tc.model, nil)
				provider.SetDefaultOptions(&config.Config{MaxTokens: maxTokens})

				body, err := provider.PrepareRequest("hello", nil)
				assert.NoError(t, err)

				var req map[string]interface{}
				assert.NoError(t, json.Unmarshal(body, &req))

				_, legacySent := req["max_tokens"]
				assert.False(t, legacySent, "the cap should never go out under its legacy name")

				capValue, sent := req["max_completion_tokens"]
				if maxTokens == 0 {
					assert.False(t, sent, "no output cap should be sent")
					continue
				}
				assert.True(t, sent, "the output cap should be sent")
				assert.EqualValues(t, maxTokens, capValue, "the output cap should carry the configured value")
			}
		})
	}
}
