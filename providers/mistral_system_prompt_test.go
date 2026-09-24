package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/config"
)

// The Mistral API rejects unknown body fields, so a system prompt must travel
// as a system-role message. Sent as a field of its own it fails the whole
// request with 422 extra_forbidden.
func TestMistralPrepareRequestCarriesTheSystemPromptAsAMessage(t *testing.T) {
	for _, tc := range []struct {
		name    string
		options map[string]interface{}
		defs    map[string]interface{}
	}{
		{name: "per call", options: map[string]interface{}{"system_prompt": "SYSTEM"}},
		{name: "provider default", options: map[string]interface{}{}, defs: map[string]interface{}{"system_prompt": "SYSTEM"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := NewMistralProvider("key", "mistral-large-2512", nil)
			for k, v := range tc.defs {
				p.SetOption(k, v)
			}

			raw, err := p.PrepareRequest("USER", tc.options)
			if err != nil {
				t.Fatalf("PrepareRequest: %v", err)
			}

			var body map[string]interface{}
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatalf("unmarshalling request: %v", err)
			}
			if _, ok := body["system_prompt"]; ok {
				t.Error("system_prompt must not be a body field: the API rejects unknown fields")
			}

			messages, ok := body["messages"].([]interface{})
			if !ok || len(messages) != 2 {
				t.Fatalf("expected a system and a user message, got %v", body["messages"])
			}
			system := messages[0].(map[string]interface{})
			if system["role"] != "system" || system["content"] != "SYSTEM" {
				t.Errorf("first message = %v, want the system prompt", system)
			}
			user := messages[1].(map[string]interface{})
			if user["role"] != "user" || user["content"] != "USER" {
				t.Errorf("second message = %v, want the user prompt", user)
			}
		})
	}
}

func TestMistralPrepareRequestWithoutSystemPromptSendsOneMessage(t *testing.T) {
	p := NewMistralProvider("key", "mistral-large-2512", nil)
	p.SetDefaultOptions(&config.Config{MaxTokens: 100})

	raw, err := p.PrepareRequest("USER", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest: %v", err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshalling request: %v", err)
	}
	if messages, ok := body["messages"].([]interface{}); !ok || len(messages) != 1 {
		t.Fatalf("expected the user message alone, got %v", body["messages"])
	}
}

func TestMistralPrepareRequestWithSchemaCarriesTheSystemPromptAsAMessage(t *testing.T) {
	p := NewMistralProvider("key", "mistral-large-2512", nil)

	raw, err := p.PrepareRequestWithSchema("USER", map[string]interface{}{"system_prompt": "SYSTEM"}, map[string]any{"type": "object"})
	if err != nil {
		t.Fatalf("PrepareRequestWithSchema: %v", err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshalling request: %v", err)
	}
	if _, ok := body["system_prompt"]; ok {
		t.Error("system_prompt must not be a body field on the schema path either")
	}
	messages, ok := body["messages"].([]interface{})
	if !ok || len(messages) != 2 {
		t.Fatalf("expected a system and a user message, got %v", body["messages"])
	}
}
