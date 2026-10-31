// Package providers implements LLM provider interfaces and implementations.
package providers

import "strings"

// chatCompletionsURL normalizes a configured endpoint into a chat completions
// URL, so an OpenAI-compatible service can be named by its host, by its
// versioned base URL, or by the full path.
// Accepts: http://host:port, http://host:port/v1, http://host:port/v1/chat/completions
func chatCompletionsURL(endpoint string) string {
	endpoint = strings.TrimSuffix(endpoint, "/")
	if strings.HasSuffix(endpoint, "/chat/completions") {
		return endpoint
	}
	if !strings.HasSuffix(endpoint, "/v1") {
		endpoint += "/v1"
	}
	return endpoint + "/chat/completions"
}

// messagesURL normalizes a configured endpoint into an Anthropic Messages API
// URL the same way: host, versioned base URL, or full path.
// Accepts: http://host:port, http://host:port/v1, http://host:port/v1/messages
func messagesURL(endpoint string) string {
	endpoint = strings.TrimSuffix(endpoint, "/")
	if strings.HasSuffix(endpoint, "/messages") {
		return endpoint
	}
	if !strings.HasSuffix(endpoint, "/v1") {
		endpoint += "/v1"
	}
	return endpoint + "/messages"
}
