package main

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// RetrieveRequest is the request body for context retrieval.
type RetrieveRequest struct {
	Query    string   `json:"query" binding:"required"`
	UserID   string   `json:"user_id"`
	TenantID string   `json:"tenant_id"`
	ScopeIDs []string `json:"scope_ids"`
	Limit    int      `json:"limit"`
	Language string   `json:"language"`
}

// RetrieveResult is the response for context retrieval.
type RetrieveResult struct {
	Results []RetrievalItem `json:"results"`
	Total   int             `json:"total"`
}

// RetrievalItem is a single retrieved document.
type RetrievalItem struct {
	EvidenceID   string  `json:"evidence_id"`
	Content      string  `json:"content"`
	Score        float64 `json:"score"`
	FTSScore     float64 `json:"fts_score"`
	RecencyScore float64 `json:"recency_score"`
	VectorScore  float64 `json:"vector_score"`
}

// retrieveContext handles POST /api/v1/context/retrieve.
//
// In production, this would query a vector database (e.g. Qdrant, Weaviate)
// and a full-text search index, then fuse the results.
func retrieveContext(c *gin.Context) {
	var req RetrieveRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	if req.Limit <= 0 {
		req.Limit = 10
	}
	if req.Limit > 50 {
		req.Limit = 50
	}
	if len(req.Query) > 10000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query too long (max 10000 chars)"})
		return
	}
	if len(req.ScopeIDs) > 100 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "too many scope_ids (max 100)"})
		return
	}

	// In production: query vector DB + FTS index
	// For now, return empty results
	result := &RetrieveResult{
		Results: []RetrievalItem{},
		Total:   0,
	}

	c.JSON(http.StatusOK, result)
}

// GenerationRequest is the request body for text generation.
type GenerationRequest struct {
	Prompt      string             `json:"prompt" binding:"required"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature"`
	Grammar     *GrammarConstraint `json:"grammar"`
	ModelID     string             `json:"model_id"`
	Stream      bool               `json:"stream"`
	UserID      string             `json:"user_id"`
	TenantID    string             `json:"tenant_id"`
	LoRAAdapter string             `json:"lora_adapter"`
}

// GrammarConstraint specifies output format constraints.
type GrammarConstraint struct {
	Type    string      `json:"type"` // json_schema, regex, lark
	Schema  interface{} `json:"schema"`
	Pattern string      `json:"pattern"`
	Grammar string      `json:"grammar"`
}

// GenerationResult is the response for text generation.
type GenerationResult struct {
	Text             string  `json:"text"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TTFTMS           int64   `json:"ttft_ms"`
	TotalMS          int64   `json:"total_ms"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
	GrammarValid     bool    `json:"grammar_valid"`
	ModelID          string  `json:"model_id"`
}

// generate handles POST /api/v1/generation/generate.
//
// In production, this would forward to a llama.cpp server or cloud LLM API.
// The grammar constraint is applied via GBNF (llama.cpp) or function calling.
func generate(c *gin.Context) {
	var req GenerationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	if len(req.Prompt) > 32000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt too long (max 32000 chars)"})
		return
	}

	// Validate grammar constraint if provided
	if req.Grammar != nil {
		switch req.Grammar.Type {
		case "json_schema":
			if req.Grammar.Schema == nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "schema required for json_schema grammar type"})
				return
			}
		case "regex":
			if req.Grammar.Pattern == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "pattern required for regex grammar type"})
				return
			}
		case "lark":
			if req.Grammar.Grammar == "" {
				c.JSON(http.StatusBadRequest, gin.H{"error": "grammar required for lark grammar type"})
				return
			}
		case "":
			// No grammar type — ignore
		default:
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid grammar type: " + req.Grammar.Type})
			return
		}
	}

	// Validate model_id against allowlist if provided
	if req.ModelID != "" {
		validModels := map[string]bool{
			"qwen3.5-0.8b-q4":            true,
			"multilingual-e5-small-int8": true,
			"safety-classifier-int8":     true,
		}
		if !validModels[req.ModelID] {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid model_id"})
			return
		}
	}

	if req.MaxTokens == 0 {
		req.MaxTokens = 256
	}
	if req.MaxTokens > 2048 {
		req.MaxTokens = 2048
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}

	// In production: forward to llama.cpp server or cloud LLM
	// For now, return a placeholder
	start := time.Now()
	result := &GenerationResult{
		Text:             "[server-side offload placeholder]",
		PromptTokens:     len(req.Prompt) / 4,
		CompletionTokens: 1,
		TTFTMS:           time.Since(start).Milliseconds(),
		TotalMS:          time.Since(start).Milliseconds(),
		TokensPerSecond:  0,
		GrammarValid:     true,
		ModelID:          req.ModelID,
	}

	c.JSON(http.StatusOK, result)
}

// ActionValidationRequest is the request body for action validation.
type ActionValidationRequest struct {
	ToolPlan interface{} `json:"tool_plan"`
	UserID   string      `json:"user_id"`
	TenantID string      `json:"tenant_id"`
}

// ActionValidationResult is the response for action validation.
type ActionValidationResult struct {
	Valid     bool   `json:"valid"`
	StepCount int    `json:"step_count"`
	Error     string `json:"error,omitempty"`
}

// validateAction handles POST /api/v1/action/validate.
func validateAction(c *gin.Context) {
	var req ActionValidationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	// In production: forward to Rust action plane
	result := &ActionValidationResult{
		Valid:     true,
		StepCount: 0,
	}

	c.JSON(http.StatusOK, result)
}

// ModelInfo describes an available model.
type ModelInfo struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Type         string   `json:"type"`
	SizeBytes    int64    `json:"size_bytes"`
	Quantization string   `json:"quantization"`
	Capabilities []string `json:"capabilities"`
	Languages    []string `json:"languages"`
	MinTier      string   `json:"min_tier"`
}

// listModels handles GET /api/v1/models.
func listModels(c *gin.Context) {
	models := []ModelInfo{
		{
			ID:           "qwen3.5-0.8b-q4",
			Name:         "Qwen3.5 0.8B Q4_K_M",
			Version:      "1.0.0",
			Type:         "generative",
			SizeBytes:    500000000,
			Quantization: "Q4_K_M",
			Capabilities: []string{"summarize", "translate", "generate"},
			Languages:    []string{"en", "vi", "zh", "ja", "ko", "es"},
			MinTier:      "high",
		},
		{
			ID:           "multilingual-e5-small-int8",
			Name:         "Multilingual E5 Small INT8",
			Version:      "1.0.0",
			Type:         "embedding",
			SizeBytes:    45000000,
			Quantization: "INT8",
			Capabilities: []string{"embed"},
			Languages:    []string{"en", "vi", "zh", "ja", "ko", "es"},
			MinTier:      "medium",
		},
		{
			ID:           "safety-classifier-int8",
			Name:         "Safety Classifier INT8",
			Version:      "1.0.0",
			Type:         "safety",
			SizeBytes:    25000000,
			Quantization: "INT8",
			Capabilities: []string{"classify"},
			Languages:    []string{"en", "vi", "zh", "ja", "ko", "es"},
			MinTier:      "medium",
		},
	}

	c.JSON(http.StatusOK, gin.H{"models": models})
}
