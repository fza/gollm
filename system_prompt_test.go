package gollm_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/teilomillet/gollm"
)

// wireProvider names a provider whose request carries the system prompt in a
// slot of its own, and how to read that slot, the user content and the
// response schema back. Its response reports usage as 11 in, 22 out.
type wireProvider struct {
	name     string
	opts     func(url string) []gollm.ConfigOption
	response func(text string) string
	system   func(t *testing.T, body map[string]any) string
	user     func(t *testing.T, body map[string]any) string
	schema   func(t *testing.T, body map[string]any) any
}

var wireProviders = []wireProvider{
	{
		name: "openai",
		opts: func(url string) []gollm.ConfigOption {
			return []gollm.ConfigOption{gollm.SetProvider("openai"), gollm.SetOpenAIEndpoint(url)}
		},
		response: func(text string) string {
			return `{"choices":[{"message":{"content":` + jsonString(text) + `}}],"usage":{"prompt_tokens":11,"completion_tokens":22}}`
		},
		system: func(t *testing.T, body map[string]any) string {
			return messageContent(t, body, "system", "developer")
		},
		user: func(t *testing.T, body map[string]any) string {
			return messageContent(t, body, "user")
		},
		schema: func(t *testing.T, body map[string]any) any {
			format := asMap(t, body["response_format"])
			if format["type"] != "json_schema" {
				t.Errorf("response_format.type = %v, want json_schema", format["type"])
			}
			js := asMap(t, format["json_schema"])
			if js["strict"] != true {
				t.Errorf("json_schema.strict = %v, want true", js["strict"])
			}
			return js["schema"]
		},
	},
	{
		name: "anthropic",
		opts: func(url string) []gollm.ConfigOption {
			return []gollm.ConfigOption{gollm.SetProvider("anthropic"), gollm.SetAnthropicEndpoint(url)}
		},
		response: func(text string) string {
			return `{"content":[{"type":"text","text":` + jsonString(text) + `}],"usage":{"input_tokens":11,"output_tokens":22}}`
		},
		schema: func(t *testing.T, body map[string]any) any {
			format := asMap(t, asMap(t, body["output_config"])["format"])
			if format["type"] != "json_schema" {
				t.Errorf("output_config.format.type = %v, want json_schema", format["type"])
			}
			return format["schema"]
		},
		system: func(t *testing.T, body map[string]any) string {
			var parts []string
			for _, block := range asSlice(t, body["system"]) {
				parts = append(parts, asMap(t, block)["text"].(string))
			}
			return strings.Join(parts, "\n\n")
		},
		user: func(t *testing.T, body map[string]any) string {
			msgs := asSlice(t, body["messages"])
			if len(msgs) != 1 {
				t.Fatalf("want one user message, got %d", len(msgs))
			}
			var parts []string
			for _, block := range asSlice(t, asMap(t, msgs[0])["content"]) {
				parts = append(parts, asMap(t, block)["text"].(string))
			}
			return strings.Join(parts, "")
		},
	},
	{
		name: "ollama",
		opts: func(url string) []gollm.ConfigOption {
			return []gollm.ConfigOption{gollm.SetProvider("ollama"), gollm.SetOllamaEndpoint(url)}
		},
		response: func(text string) string {
			return `{"model":"m","response":` + jsonString(text) + `,"done":true,"prompt_eval_count":11,"eval_count":22}`
		},
		system: func(t *testing.T, body map[string]any) string {
			s, _ := body["system"].(string)
			return s
		},
		user: func(t *testing.T, body map[string]any) string {
			s, _ := body["prompt"].(string)
			return s
		},
		schema: func(t *testing.T, body map[string]any) any {
			return body["format"]
		},
	},
}

// wireServer serves the provider's response to every generation request and
// answers Ollama's endpoint probe, which is not one.
func wireServer(t *testing.T, handle func(w http.ResponseWriter, raw []byte)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = w.Write([]byte(`{"models":[]}`))
			return
		}
		raw, _ := io.ReadAll(r.Body)
		handle(w, raw)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func newWireLLM(t *testing.T, p wireProvider, url string, extra ...gollm.ConfigOption) gollm.LLM {
	t.Helper()
	opts := append(p.opts(url),
		gollm.SetModel("m"), gollm.SetAPIKey("gateway-issued-key-aaaaaaaa"),
		gollm.SetMaxRetries(0), gollm.SetLogLevel(gollm.LogLevelOff))
	llm, err := gollm.NewLLM(append(opts, extra...)...)
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	return llm
}

// The system prompt goes to the provider's system slot and nowhere else: a
// copy in the user content doubles its tokens and puts it after the cacheable
// prefix.
func TestSystemPromptSentOnceInSystemSlot(t *testing.T) {
	for _, p := range wireProviders {
		t.Run(p.name, func(t *testing.T) {
			var raw []byte
			srv := wireServer(t, func(w http.ResponseWriter, body []byte) {
				raw = body
				_, _ = w.Write([]byte(p.response("ok")))
			})

			llm := newWireLLM(t, p, srv.URL)
			prompt := gollm.NewPrompt("USER-MARK", gollm.WithSystemPrompt("SYS-MARK", gollm.CacheTypeEphemeral))
			if _, err := llm.GenerateWithUsage(context.Background(), prompt); err != nil {
				t.Fatalf("GenerateWithUsage: %v", err)
			}

			if n := strings.Count(string(raw), "SYS-MARK"); n != 1 {
				t.Errorf("system prompt sent %d times, want once: %s", n, raw)
			}
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatalf("request body: %v", err)
			}
			if got := p.system(t, body); got != "SYS-MARK" {
				t.Errorf("system slot = %q, want SYS-MARK", got)
			}
			if got := p.user(t, body); got != "USER-MARK" {
				t.Errorf("user content = %q, want USER-MARK", got)
			}
		})
	}
}

// Each call sends its own system prompt even while another call with a
// different one overlaps it. The first call's first attempt is held until the
// second call's request arrives, then fails retryably, so its retry runs after
// the second call has started: a system prompt kept in client-wide state
// would leak into that retry. Not a data race the detector sees — the shared
// state is mutex-guarded, the last writer simply wins.
func TestOverlappingCallsKeepTheirSystemPrompts(t *testing.T) {
	for _, p := range wireProviders {
		t.Run(p.name, func(t *testing.T) {
			var mu sync.Mutex
			seen := map[string][]string{} // user marker -> system slots sent
			secondArrived := make(chan struct{})
			var heldFirst bool

			srv := wireServer(t, func(w http.ResponseWriter, raw []byte) {
				var body map[string]any
				if err := json.Unmarshal(raw, &body); err != nil {
					t.Errorf("request body: %v", err)
					return
				}
				user := "USER-B"
				if strings.Contains(p.user(t, body), "USER-A") {
					user = "USER-A"
				}
				mu.Lock()
				seen[user] = append(seen[user], p.system(t, body))
				hold := user == "USER-A" && !heldFirst
				if hold {
					heldFirst = true
				}
				mu.Unlock()

				switch {
				case hold:
					select {
					case <-secondArrived:
					case <-time.After(5 * time.Second):
						t.Error("second call never arrived")
					}
					w.WriteHeader(http.StatusServiceUnavailable)
				case user == "USER-B":
					close(secondArrived)
					_, _ = w.Write([]byte(p.response("ok")))
				default:
					_, _ = w.Write([]byte(p.response("ok")))
				}
			})

			llm := newWireLLM(t, p, srv.URL, gollm.SetMaxRetries(1), gollm.SetRetryDelay(time.Millisecond))

			var wg sync.WaitGroup
			run := func(user, system string) {
				defer wg.Done()
				prompt := gollm.NewPrompt(user, gollm.WithSystemPrompt(system, gollm.CacheTypeEphemeral))
				if _, err := llm.GenerateWithUsage(context.Background(), prompt); err != nil {
					t.Errorf("%s: %v", user, err)
				}
			}
			wg.Add(1)
			go run("USER-A", "SYS-A")
			// Start the second call only once the first one is held.
			for {
				mu.Lock()
				held := heldFirst
				mu.Unlock()
				if held {
					break
				}
				time.Sleep(time.Millisecond)
			}
			wg.Add(1)
			go run("USER-B", "SYS-B")
			wg.Wait()

			mu.Lock()
			defer mu.Unlock()
			for user, want := range map[string]string{"USER-A": "SYS-A", "USER-B": "SYS-B"} {
				if len(seen[user]) == 0 {
					t.Errorf("%s never reached the server", user)
				}
				for i, got := range seen[user] {
					if !strings.Contains(got, want) || strings.Contains(got, otherSystem(want)) {
						t.Errorf("%s attempt %d carried system %q, want %s", user, i+1, got, want)
					}
				}
			}
		})
	}
}

func otherSystem(s string) string {
	if s == "SYS-A" {
		return "SYS-B"
	}
	return "SYS-A"
}

func messageContent(t *testing.T, body map[string]any, roles ...string) string {
	t.Helper()
	var parts []string
	for _, m := range asSlice(t, body["messages"]) {
		msg := asMap(t, m)
		for _, role := range roles {
			if msg["role"] == role {
				parts = append(parts, msg["content"].(string))
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

func asSlice(t *testing.T, v any) []any {
	t.Helper()
	s, ok := v.([]any)
	if !ok {
		t.Fatalf("want a JSON array, got %T", v)
	}
	return s
}

func asMap(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("want a JSON object, got %T", v)
	}
	return m
}
