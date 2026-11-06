package llm

import "testing"

func TestUsageFromResponseMap(t *testing.T) {
	// No usage object → nil.
	if usageFromResponseMap(map[string]interface{}{}) != nil {
		t.Error("missing usage should yield nil")
	}

	// Anthropic-style: input/output plus the prompt-cache breakdown. JSON
	// numbers decode to float64, which the parser must coerce to int.
	u := usageFromResponseMap(map[string]interface{}{
		"usage": map[string]interface{}{
			"input_tokens":                float64(6341),
			"output_tokens":               float64(8),
			"cache_read_input_tokens":     float64(6163),
			"cache_creation_input_tokens": float64(0),
		},
	})
	if u == nil {
		t.Fatal("expected usage")
	}
	if u.InputTokens != 6341 || u.OutputTokens != 8 || u.CacheReadInputTokens != 6163 {
		t.Errorf("anthropic parse wrong: %+v", u)
	}

	// OpenAI-style: prompt/completion/total.
	u2 := usageFromResponseMap(map[string]interface{}{
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(100),
			"completion_tokens": float64(20),
			"total_tokens":      float64(120),
		},
	})
	if u2 == nil {
		t.Fatal("expected usage")
	}
	if u2.PromptTokens != 100 || u2.CompletionTokens != 20 || u2.TotalTokens != 120 {
		t.Errorf("openai parse wrong: %+v", u2)
	}

	// OpenAI cache hits: nested under prompt_tokens_details.cached_tokens.
	u2c := usageFromResponseMap(map[string]interface{}{
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(10248),
			"completion_tokens": float64(56),
			"total_tokens":      float64(10304),
			"prompt_tokens_details": map[string]interface{}{
				"cached_tokens": float64(9984),
			},
		},
	})
	if u2c == nil {
		t.Fatal("expected usage")
	}
	if u2c.CacheReadInputTokens != 9984 {
		t.Errorf("openai cached_tokens not read as cache reads: %+v", u2c)
	}

	// A gateway that reports the call's dollar cost (Opper) puts it in the
	// usage object; a provider that reports none leaves it zero.
	u2o := usageFromResponseMap(map[string]interface{}{
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(18),
			"completion_tokens": float64(21),
			"cost":              1.2534984e-05,
		},
	})
	if u2o == nil || u2o.CostUSD != 1.2534984e-05 {
		t.Errorf("reported cost not read: %+v", u2o)
	}
	if u2.CostUSD != 0 {
		t.Errorf("cost invented where none was reported: %+v", u2)
	}

	// Ollama: counts at the top level of the final streamed object, with no
	// usage object anywhere.
	u3 := usageFromResponseMap(map[string]interface{}{
		"model":             "glm-5.3-flash:cloud",
		"done":              true,
		"prompt_eval_count": float64(4210),
		"eval_count":        float64(96),
	})
	if u3 == nil {
		t.Fatal("expected usage")
	}
	if u3.PromptTokens != 4210 || u3.CompletionTokens != 96 || u3.TotalTokens != 4306 {
		t.Errorf("ollama parse wrong: %+v", u3)
	}

	// A response with neither shape still reports nothing.
	if usageFromResponseMap(map[string]interface{}{"done": true}) != nil {
		t.Error("a response with no counts should yield nil")
	}
}
