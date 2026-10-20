package llm

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/teilomillet/gollm/providers"
)

func newOpenAITestLLM(rt http.RoundTripper) *LLMImpl {
	l := newTestLLM(rt)
	l.Provider = providers.NewOpenAIProvider("fake-api-key", "glm-5.3-flash", nil)
	return l
}

func TestGenerateWithUsageReportsWhyNoAnswerCame(t *testing.T) {
	testCases := []struct {
		description string
		body        string
		truncated   bool
	}{
		{
			description: "reasoning spent the cap",
			body:        `{"choices":[{"message":{"content":""},"finish_reason":"length"}],"usage":{"prompt_tokens":900,"completion_tokens":8192,"completion_tokens_details":{"reasoning_tokens":8192}}}`,
			truncated:   true,
		},
		{
			description: "model stopped without answering",
			body:        `{"choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":900,"completion_tokens":4000,"completion_tokens_details":{"reasoning_tokens":4000}}}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			rt := &scriptedRT{steps: []func() (*http.Response, error){resp(200, tc.body)}}

			_, err := newOpenAITestLLM(rt).GenerateWithUsage(context.Background(), &Prompt{Input: "hi"})

			var noAnswer *NoAnswerError
			if !errors.As(err, &noAnswer) {
				t.Fatalf("an empty answer should surface as a NoAnswerError, got %v", err)
			}
			if noAnswer.Truncated() != tc.truncated {
				t.Errorf("truncation should follow the finish reason, got %v for %q", noAnswer.Truncated(), noAnswer.FinishReason)
			}
			if noAnswer.Usage == nil || noAnswer.Usage.ReasoningTokens == 0 {
				t.Errorf("the reasoning spent should be carried, got %+v", noAnswer.Usage)
			}
			if rt.calls != 1 {
				t.Errorf("a delivered response should not be retried, made %d calls", rt.calls)
			}
		})
	}
}

func TestGenerateWithUsageCarriesFinishReason(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		resp(200, `{"choices":[{"message":{"content":"{\"findings\":["},"finish_reason":"length"}],"usage":{"completion_tokens":8192,"completion_tokens_details":{"reasoning_tokens":7602}}}`),
	}}

	got, err := newOpenAITestLLM(rt).GenerateWithUsage(context.Background(), &Prompt{Input: "hi"})
	if err != nil {
		t.Fatalf("a cut answer with content should still be returned, got %v", err)
	}
	if !got.Truncated() {
		t.Error("an answer ended at the cap should report itself truncated")
	}
	if got.Usage.ReasoningTokens != 7602 {
		t.Errorf("the reasoning tokens should be read from completion_tokens_details, got %d", got.Usage.ReasoningTokens)
	}
}

func TestFinishReasonFromResponseMap(t *testing.T) {
	testCases := []struct {
		description string
		response    map[string]interface{}
		want        string
	}{
		{"openai", map[string]interface{}{"choices": []interface{}{map[string]interface{}{"finish_reason": "length"}}}, "length"},
		{"anthropic", map[string]interface{}{"stop_reason": "max_tokens"}, "max_tokens"},
		{"ollama", map[string]interface{}{"done_reason": "stop"}, "stop"},
		{"none reported", map[string]interface{}{"choices": []interface{}{}}, ""},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			if got := finishReasonFromResponseMap(tc.response); got != tc.want {
				t.Errorf("finish reason should be read from the provider's own field, got %q", got)
			}
		})
	}
}

func TestGenerateWithUsageCarriesBillingCost(t *testing.T) {
	testCases := []struct {
		description string
		body        string
		want        string
	}{
		{
			description: "billing cost kept verbatim",
			body:        `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":21},"billing_cost":{"energy":"0.103","credits":"0.0000103","paid_with":"energy"}}`,
			want:        `{"energy":"0.103","credits":"0.0000103","paid_with":"energy"}`,
		},
		{
			description: "billing cost without a usage object",
			body:        `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"billing_cost":{"energy":1.5e-3}}`,
			want:        `{"energy":1.5e-3}`,
		},
		{
			description: "no billing cost",
			body:        `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":19}}`,
		},
		{
			description: "billing cost that is not an object",
			body:        `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":19},"billing_cost":null}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			rt := &scriptedRT{steps: []func() (*http.Response, error){resp(200, tc.body)}}

			got, err := newOpenAITestLLM(rt).GenerateWithUsage(context.Background(), &Prompt{Input: "hi"})
			if err != nil {
				t.Fatal(err)
			}
			var billingCost string
			if got.Usage != nil {
				billingCost = string(got.Usage.BillingCost)
			}
			if billingCost != tc.want {
				t.Errorf("the billing cost should be the provider's object as sent, got %q", billingCost)
			}
		})
	}
}

func TestGenerateWithUsageCarriesBillingCostOnNoAnswer(t *testing.T) {
	rt := &scriptedRT{steps: []func() (*http.Response, error){
		resp(200, `{"choices":[{"message":{"content":""},"finish_reason":"length"}],"usage":{"completion_tokens":64},"billing_cost":{"energy":"0.2"}}`),
	}}

	_, err := newOpenAITestLLM(rt).GenerateWithUsage(context.Background(), &Prompt{Input: "hi"})

	var noAnswer *NoAnswerError
	if !errors.As(err, &noAnswer) {
		t.Fatalf("an empty answer should surface as a NoAnswerError, got %v", err)
	}
	if string(noAnswer.Usage.BillingCost) != `{"energy":"0.2"}` {
		t.Errorf("a billed reply without an answer should still carry its cost, got %q", noAnswer.Usage.BillingCost)
	}
}
