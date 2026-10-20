package config_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm/config"
)

func TestMaxTokensOptions(t *testing.T) {
	testCases := []struct {
		description string
		option      config.ConfigOption
		want        int
	}{
		{"no cap leaves MaxTokens zero", config.SetNoMaxTokens(), 0},
		{"a zero cap is raised to the smallest valid one", config.SetMaxTokens(0), 1},
		{"a positive cap is kept", config.SetMaxTokens(2048), 2048},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			cfg := &config.Config{MaxTokens: 4096}
			tc.option(cfg)
			assert.Equal(t, tc.want, cfg.MaxTokens, "MaxTokens should follow the option applied")
		})
	}
}
