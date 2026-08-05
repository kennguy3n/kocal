// Package main implements the KChat server-side offload service.
//
// This Go service handles AI inference requests when the on-device runtime
// cannot process them (low-tier device, thermal throttling, battery saver,
// or model not installed). It provides:
//
//   - Safety classification (forwards to deterministic rules + optional ML)
//   - Context retrieval (forwards to a vector DB or search service)
//   - Generation (forwards to llama.cpp server or cloud LLM API)
//   - Action validation (forwards to the Rust action plane)
//
// The service is designed to be stateless and horizontally scalable.
// All requests are authenticated with a bearer token and rate-limited.
//
// Routes:
//
//	POST /api/v1/safety/classify    - Classify text for safety
//	POST /api/v1/context/retrieve   - Retrieve context documents
//	POST /api/v1/generation/generate - Generate text with grammar constraints
//	POST /api/v1/action/validate    - Validate a tool plan
//	GET  /api/v1/health             - Health check (no auth)
//	GET  /api/v1/models             - List available models
package main

import (
	"crypto/subtle"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Configure Gin
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Read API key at startup (fail-safe: empty key = reject all unless explicitly disabled)
	apiKey := os.Getenv("KCHAT_API_KEY")
	authDisabled := os.Getenv("KCHAT_AUTH_DISABLED") == "true"

	if apiKey == "" && !authDisabled {
		log.Fatal("KCHAT_API_KEY is not set. Set KCHAT_AUTH_DISABLED=true to bypass auth (dev only).")
	}

	rl := newRateLimiter(100, time.Minute)

	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Limit request body size to 1MB to prevent body-bomb DoS
	r.Use(func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1<<20)
		c.Next()
	})

	// Health check — no auth, no rate limit
	r.GET("/api/v1/health", healthCheck)

	// Auth + rate limit middleware applied only to API group
	v1 := r.Group("/api/v1")
	v1.Use(authMiddleware(apiKey, authDisabled))
	v1.Use(rateLimitMiddleware(rl))
	{
		v1.POST("/safety/classify", classifySafety)
		v1.POST("/context/retrieve", retrieveContext)
		v1.POST("/generation/generate", generate)
		v1.POST("/action/validate", validateAction)
		v1.GET("/models", listModels)
	}

	log.Printf("KChat server offload service starting on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// healthCheck returns service health status.
func healthCheck(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":     "healthy",
		"service":    "kchat-server-offload",
		"version":    "1.0.0",
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
		"uptime_sec": int64(time.Since(startTime).Seconds()),
	})
}

var startTime = time.Now()

// authMiddleware validates the bearer token using constant-time comparison.
// Fail-safe: if apiKey is empty and auth is not explicitly disabled, reject all.
func authMiddleware(apiKey string, authDisabled bool) gin.HandlerFunc {
	return func(c *gin.Context) {
		if authDisabled {
			c.Next()
			return
		}

		token := c.GetHeader("Authorization")
		if token == "" {
			c.JSON(401, gin.H{"error": "unauthorized"})
			c.Abort()
			return
		}
		expected := "Bearer " + apiKey

		// Constant-time comparison to prevent timing attacks
		if subtle.ConstantTimeCompare([]byte(token), []byte(expected)) != 1 {
			c.JSON(401, gin.H{"error": "unauthorized"})
			c.Abort()
			return
		}
		c.Next()
	}
}

// rateLimiter is a thread-safe rate limiter per IP with background cleanup.
type rateLimiter struct {
	mu       sync.Mutex
	requests map[string][]time.Time
	maxReqs  int
	window   time.Duration
}

func newRateLimiter(maxReqs int, window time.Duration) *rateLimiter {
	rl := &rateLimiter{
		requests: make(map[string][]time.Time),
		maxReqs:  maxReqs,
		window:   window,
	}
	// Start background cleanup goroutine to avoid blocking request path
	go rl.cleanupLoop()
	return rl
}

// cleanupLoop periodically removes stale entries in the background.
func (rl *rateLimiter) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		rl.cleanup()
	}
}

// cleanup removes all stale IP entries. Called from background goroutine.
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	cutoff := time.Now().Add(-rl.window)
	for k, times := range rl.requests {
		var recent []time.Time
		for _, t := range times {
			if t.After(cutoff) {
				recent = append(recent, t)
			}
		}
		if len(recent) == 0 {
			delete(rl.requests, k)
		} else {
			rl.requests[k] = recent
		}
	}
}

// allow checks if an IP is within the rate limit.
func (rl *rateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Clean current IP's old entries
	var recent []time.Time
	for _, t := range rl.requests[ip] {
		if t.After(cutoff) {
			recent = append(recent, t)
		}
	}

	if len(recent) >= rl.maxReqs {
		rl.requests[ip] = recent
		return false
	}

	recent = append(recent, now)
	rl.requests[ip] = recent
	return true
}

// rateLimitMiddleware enforces per-IP rate limiting using a thread-safe limiter.
func rateLimitMiddleware(rl *rateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := c.ClientIP()
		if !rl.allow(ip) {
			c.JSON(429, gin.H{"error": "rate limit exceeded"})
			c.Abort()
			return
		}
		c.Next()
	}
}
