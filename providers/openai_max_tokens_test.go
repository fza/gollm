package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/teilomillet/gollm/config"
)

// TestNoOutputCapWhenMaxTokensUnset verifies that a zero MaxTokens sends no
// output cap on OpenAI, so a reasoning model keeps its own output limit, while
// a positive one is sent as the cap. The cap goes out as max_completion_tokens
// whatever the model.
func TestNoOutputCapWhenMaxTokensUnset(t *testing.T) {
	testCases := []struct {
		description string
		model       string
	}{
		{"non-reasoning model", "glm-5.3-flash"},
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
