package llm

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

func TestRequestPathsCarryProviderErrorDetail(t *testing.T) {
	testCases := []struct {
		description string
		request     func(l *LLMImpl) error
	}{
		{
			description: "schema request",
			request: func(l *LLMImpl) error {
				_, err := l.GenerateWithSchema(context.Background(), &Prompt{Input: "hi"}, map[string]interface{}{"type": "object"})
				return err
			},
		},
		{
			description: "stream request",
			request: func(l *LLMImpl) error {
				_, err := l.Stream(context.Background(), &Prompt{Input: "hi"})
				return err
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			rt := &scriptedRT{steps: []func() (*http.Response, error){
				resp(400, `{"error":{"message":"The model does not support JSON schema mode","type":"invalid_request_error","code":"model_not_capable"}}`),
			}}
			l := newTestLLM(rt)
			l.Provider = providers.NewOpenAIProvider("fake-api-key", "glm-5.3-flash", nil)
			l.MaxRetries = 0
			logger := &recordingLogger{Logger: utils.NewLogger(utils.LogLevelOff)}
			l.logger = logger

			err := tc.request(l)

			var llmErr *LLMError
			if !errors.As(err, &llmErr) {
				t.Fatalf("the rejection should surface as an LLMError, got %v", err)
			}
			if llmErr.ProviderCode != "model_not_capable" || llmErr.ProviderType != "invalid_request_error" || llmErr.ProviderMessage == "" {
				t.Errorf("the provider's error detail should be carried, got %+v", llmErr)
			}
			if llmErr.Retryable {
				t.Error("a 400 rejection should not be retryable")
			}
			if len(logger.errors) != 0 {
				t.Errorf("a returned permanent error should not also be logged as an error, got %v", logger.errors)
			}
		})
	}
}
