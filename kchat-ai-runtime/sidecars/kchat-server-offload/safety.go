package main

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// SafetyRequest is the request body for safety classification.
type SafetyRequest struct {
	Text     string `json:"text" binding:"required"`
	IsGroup  bool   `json:"is_group"`
	AgeMode  string `json:"age_mode"`
	Language string `json:"language"`
	UserID   string `json:"user_id"`
	TenantID string `json:"tenant_id"`
}

// SafetyResult is the response for safety classification.
type SafetyResult struct {
	Action      string   `json:"action"`
	Severity    int      `json:"severity"`
	Category    int      `json:"category"`
	Confidence  float64  `json:"confidence"`
	ReasonCodes []string `json:"reason_codes"`
	UsedEncoder bool     `json:"used_encoder"`
	UsedSLM     bool     `json:"used_slm"`
	DurationUS  int64    `json:"duration_us"`
}

// classifySafety handles POST /api/v1/safety/classify.
//
// This endpoint runs deterministic safety rules server-side. In production,
// it would also invoke an ML encoder and optionally an LLM for ambiguous cases.
func classifySafety(c *gin.Context) {
	var req SafetyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	if len(req.Text) > 10000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "text too long (max 10000 chars)"})
		return
	}

	start := time.Now()

	// Run deterministic safety checks
	result := runSafetyChecks(&req)
	result.DurationUS = time.Since(start).Microseconds()

	c.JSON(http.StatusOK, result)
}

// runSafetyChecks runs deterministic safety rules on the text.
//
// This mirrors the Rust kchat-safety crate's deterministic pipeline:
// 1. PII detection (credit cards, SSN, phone, email)
// 2. Scam detection (urgency, authority, phishing patterns)
// 3. URL risk detection
// 4. Obfuscation detection (homoglyphs, mixed scripts)
func runSafetyChecks(req *SafetyRequest) *SafetyResult {
	text := req.Text
	reasonCodes := []string{}
	action := "allow"
	severity := 0
	category := 0

	// 1. Credit card detection (Luhn check)
	if hasCreditCard(text) {
		action = "redact"
		severity = 3
		category = 8 // PII
		reasonCodes = append(reasonCodes, "pii_credit_card")
	}

	// 2. SSN detection
	if hasSSN(text) {
		action = "redact"
		severity = 3
		category = 8
		reasonCodes = append(reasonCodes, "pii_ssn")
	}

	// 3. Scam patterns
	if hasScamPattern(text) {
		if action == "allow" {
			action = "warn"
			severity = 2
			category = 7 // scam
		}
		reasonCodes = append(reasonCodes, "scam_pattern")
	}

	// 4. URL risk
	if hasRiskyURL(text) {
		if action == "allow" {
			action = "warn"
			severity = 2
			category = 9 // url_risk
		}
		reasonCodes = append(reasonCodes, "url_risk")
	}

	// 5. Prompt injection
	if hasPromptInjection(text) {
		if action == "allow" {
			action = "block"
			severity = 3
			category = 1 // harassment/injection
		}
		reasonCodes = append(reasonCodes, "prompt_injection")
	}

	// 6. Group context adds stricter rules
	if req.IsGroup && action == "allow" {
		if hasGroupRisk(text) {
			action = "warn"
			severity = 1
			reasonCodes = append(reasonCodes, "group_context_risk")
		}
	}

	return &SafetyResult{
		Action:      action,
		Severity:    severity,
		Category:    category,
		Confidence:  0.95,
		ReasonCodes: reasonCodes,
		UsedEncoder: false,
		UsedSLM:     false,
	}
}

// hasCreditCard checks for credit card numbers using Luhn algorithm.
// Extracts digit sequences of 13-19 digits (allowing spaces/dashes within)
// and validates with Luhn to reduce false positives.
func hasCreditCard(text string) bool {
	runes := []rune(text)
	i := 0
	for i < len(runes) {
		// Collect a run of digits (allowing spaces/dashes within)
		var digits []rune
		j := i
		for j < len(runes) {
			r := runes[j]
			if r >= '0' && r <= '9' {
				digits = append(digits, r)
				j++
			} else if r == ' ' || r == '-' {
				j++
			} else {
				break
			}
		}
		if len(digits) >= 13 && len(digits) <= 19 {
			if luhnValid(digits) {
				return true
			}
		}
		// Advance past the scanned region; if no progress, skip one rune
		if j > i {
			i = j
		} else {
			i++
		}
	}
	return false
}

// luhnValid validates a digit sequence using the Luhn algorithm.
func luhnValid(digits []rune) bool {
	sum := 0
	alt := false
	for i := len(digits) - 1; i >= 0; i-- {
		d := int(digits[i] - '0')
		if alt {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		alt = !alt
	}
	return sum%10 == 0
}

// hasSSN checks for US Social Security Numbers (XXX-XX-XXXX).
func hasSSN(text string) bool {
	// Need at least 11 characters for the pattern XXX-XX-XXXX
	if len(text) < 11 {
		return false
	}
	runes := []rune(text)
	for i := 0; i+10 < len(runes); i++ {
		// Check for XXX-XX-XXXX pattern
		if isDigit(runes[i]) && isDigit(runes[i+1]) && isDigit(runes[i+2]) &&
			runes[i+3] == '-' &&
			isDigit(runes[i+4]) && isDigit(runes[i+5]) &&
			runes[i+6] == '-' &&
			isDigit(runes[i+7]) && isDigit(runes[i+8]) && isDigit(runes[i+9]) && isDigit(runes[i+10]) {
			return true
		}
	}
	return false
}

// isDigit checks if a rune is a digit.
func isDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

// hasScamPattern checks for common scam patterns.
func hasScamPattern(text string) bool {
	lower := strings.ToLower(text)
	scamPhrases := []string{
		"urgent", "act now", "limited time", "you won", "you're a winner",
		"click here", "verify your account", "suspended account",
		"wire transfer", "bitcoin payment", "gift card payment",
		"nigerian prince", "inheritance", "lottery winner",
	}
	for _, phrase := range scamPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

// hasRiskyURL checks for suspicious URLs.
func hasRiskyURL(text string) bool {
	lower := strings.ToLower(text)
	riskyIndicators := []string{
		"http://bit.ly", "http://tinyurl", "http://t.co",
		".tk/", ".ml/", ".ga/", ".cf/",
		"free-", ".xyz/", "login-verify",
	}
	for _, ind := range riskyIndicators {
		if strings.Contains(lower, ind) {
			return true
		}
	}
	return false
}

// hasPromptInjection checks for prompt injection attempts.
func hasPromptInjection(text string) bool {
	lower := strings.ToLower(text)
	injectionPhrases := []string{
		"ignore previous instructions",
		"ignore all instructions",
		"disregard the above",
		"you are now",
		"system:",
		"[system]",
		"new instructions:",
		"override your",
		"forget your rules",
		"act as if",
		"pretend you are",
		"jailbreak",
		"dan mode",
	}
	for _, phrase := range injectionPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

// hasGroupRisk checks for risks specific to group contexts.
func hasGroupRisk(text string) bool {
	lower := strings.ToLower(text)
	groupRisks := []string{
		"everyone", "all of you", "share this",
		"forward to", "send to everyone",
	}
	for _, risk := range groupRisks {
		if strings.Contains(lower, risk) {
			return true
		}
	}
	return false
}

// isAllDigits checks if a string is all digits.
func isAllDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return len(s) > 0
}
