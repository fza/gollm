package gollm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/llm"
)

// Properties out of alphabetical order: a schema that went through a Go map
// would come out reordered, and constrained decoding generates in schema order.
const orderedSchema = `{"type":"object","properties":{"reasoning":{"type":"string"},"axis":{"type":"string","enum":["form","pointing"]}},"required":["reasoning","axis"],"additionalProperties":false}`

// A response schema reaches each provider through its native structured
// output, byte for byte, on the regular request path: the system prompt keeps
// its own slot and the response's usage is reported.
func TestResponseSchemaOnTheRegularRequestPath(t *testing.T) {
	for _, p := range wireProviders {
		t.Run(p.name, func(t *testing.T) {
			var raw []byte
			srv := wireServer(t, func(w http.ResponseWriter, body []byte) {
				raw = body
				_, _ = w.Write([]byte(p.response(`{"reasoning":"r","axis":"form"}`)))
			})

			client := newWireLLM(t, p, srv.URL)
			prompt := gollm.NewPrompt("USER-MARK", gollm.WithSystemPrompt("SYS-MARK", gollm.CacheTypeEphemeral))
			resp, err := client.GenerateWithUsage(context.Background(), prompt, gollm.WithResponseSchema(json.RawMessage(orderedSchema)))
			if err != nil {
				t.Fatalf("GenerateWithUsage: %v", err)
			}

			if !bytes.Contains(raw, []byte(orderedSchema)) {
				t.Errorf("schema not sent byte for byte: %s", raw)
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatalf("request body: %v", err)
			}
			if got, _ := json.Marshal(p.schema(t, body)); !jsonEqual(t, got, []byte(orderedSchema)) {
				t.Errorf("schema in the provider's slot = %s, want %s", got, orderedSchema)
			}
			if got := p.system(t, body); got != "SYS-MARK" {
				t.Errorf("system slot = %q, want SYS-MARK", got)
			}
			if got := p.user(t, body); got != "USER-MARK" {
				t.Errorf("user content = %q, want USER-MARK", got)
			}
			if resp.Content != `{"reasoning":"r","axis":"form"}` {
				t.Errorf("content = %q", resp.Content)
			}
			if resp.Usage == nil {
				t.Fatal("usage not reported")
			}
			in, out := resp.Usage.InputTokens+resp.Usage.PromptTokens, resp.Usage.OutputTokens+resp.Usage.CompletionTokens
			if in != 11 || out != 22 {
				t.Errorf("usage = %+v, want 11 in, 22 out", resp.Usage)
			}
		})
	}
}

// A provider that cannot enforce a schema fails the call instead of asking for
// JSON in the prompt, before anything is sent.
func TestResponseSchemaRefusedWithoutNativeSupport(t *testing.T) {
	client, err := gollm.NewLLM(gollm.SetProvider("groq"), gollm.SetModel("m"),
		gollm.SetAPIKey("gsk-notarealkeyaaaaaaaaaaaa"), gollm.SetMaxRetries(0), gollm.SetLogLevel(gollm.LogLevelOff))
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	_, err = client.GenerateWithUsage(context.Background(), gollm.NewPrompt("hello"), gollm.WithResponseSchema(json.RawMessage(orderedSchema)))
	var llmErr *llm.LLMError
	if !errors.As(err, &llmErr) || llmErr.Type != llm.ErrorTypeUnsupported {
		t.Fatalf("want an unsupported error, got %v", err)
	}
	if !strings.Contains(err.Error(), "groq") {
		t.Errorf("error should name the provider: %v", err)
	}
}

func jsonEqual(t *testing.T, a, b []byte) bool {
	t.Helper()
	var x, y any
	if err := json.Unmarshal(a, &x); err != nil {
		t.Fatalf("decoding %s: %v", a, err)
	}
	if err := json.Unmarshal(b, &y); err != nil {
		t.Fatalf("decoding %s: %v", b, err)
	}
	ax, _ := json.Marshal(x)
	by, _ := json.Marshal(y)
	return bytes.Equal(ax, by)
}
