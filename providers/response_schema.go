package providers

import (
	"encoding/json"
	"fmt"
)

// rawSchema returns a response schema as JSON to embed in a request body. A
// json.RawMessage, []byte or string is kept as given, so the property order
// survives the request: constrained decoding generates properties in schema
// order.
func rawSchema(schema interface{}) (json.RawMessage, error) {
	var raw []byte
	switch s := schema.(type) {
	case json.RawMessage:
		raw = s
	case []byte:
		raw = s
	case string:
		raw = []byte(s)
	default:
		b, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		raw = b
	}
	if !json.Valid(raw) {
		return nil, fmt.Errorf("response schema is not valid JSON")
	}
	return json.RawMessage(raw), nil
}
