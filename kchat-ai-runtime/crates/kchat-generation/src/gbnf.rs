//! Minimal regex → GBNF translator.
//!
//! llama.cpp's GBNF grammar format has no `/regex/` literal syntax — a regex
//! embedded raw parses as literal slashes and produces wrong constraints or
//! init failures. This module translates a practical subset of regex syntax
//! into an equivalent GBNF rule set.
//!
//! Supported: literals, escapes (`\d \D \w \W \s \S \n \t \r` and escaped
//! metachars), char classes (`[...]`, `[^...]`, ranges), `.` (excludes `\n`),
//! groups `(...)` and `(?:...)`, alternation `|`, quantifiers `* + ? {n}
//! {n,} {n,m}`, and anchors `^`/`$` (dropped — the whole output is the match).
//!
//! Unsupported (rejected with an error): look-around `(?= (?! (?<= (?<!`,
//! backreferences `\1`, lazy quantifiers `*?`/`+?`/`??`/`{n,m}?`, and named
//! groups `(?<name>...)`.

/// Translate a regex pattern into a GBNF grammar string rooted at `root`.
pub fn regex_to_gbnf(pattern: &str) -> Result<String, RegexToGbnfError> {
    let mut parser = Parser::new(pattern);
    let expr = parser.parse_alt()?;
    parser.expect_end()?;
    Ok(format!("root ::= {}\n", expr))
}

/// Errors from regex → GBNF translation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RegexToGbnfError {
    #[error("unsupported regex construct: {0}")]
    Unsupported(String),
    #[error("unbalanced group or unexpected end of pattern")]
    UnexpectedEnd,
    #[error("unexpected character: {0}")]
    UnexpectedChar(char),
    #[error("invalid repeat specifier: {0}")]
    BadRepeat(String),
    #[error("invalid character class: {0}")]
    BadCharClass(String),
    #[error("invalid escape: \\{0}")]
    BadEscape(char),
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    _src: &'a str,
}

/// A parsed node renders to a GBNF expression fragment.
struct Node {
    gbnf: String,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            _src: src,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn expect_end(&self) -> Result<(), RegexToGbnfError> {
        if self.pos == self.chars.len() {
            Ok(())
        } else {
            Err(RegexToGbnfError::UnexpectedChar(self.chars[self.pos]))
        }
    }

    /// alternation := concat ('|' concat)*
    fn parse_alt(&mut self) -> Result<String, RegexToGbnfError> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some('|') {
            self.bump();
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(format!("({})", branches.join(" | ")))
        }
    }

    /// concat := repeat*
    fn parse_concat(&mut self) -> Result<String, RegexToGbnfError> {
        let mut parts = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            parts.push(self.parse_repeat()?);
        }
        if parts.is_empty() {
            // Empty branch — matches empty string. GBNF has no epsilon literal;
            // represent as optional empty group.
            Ok("(\"\")?".into())
        } else {
            Ok(parts.join(" "))
        }
    }

    /// repeat := atom quantifier?
    fn parse_repeat(&mut self) -> Result<String, RegexToGbnfError> {
        let atom = self.parse_atom()?;
        let atom = atom.gbnf;
        let Some(c) = self.peek() else {
            return Ok(atom);
        };
        match c {
            '*' | '+' | '?' => {
                self.bump();
                let quant = match c {
                    '*' => "*",
                    '+' => "+",
                    _ => "?",
                };
                self.reject_lazy()?;
                Ok(format!("{atom}{quant}"))
            }
            '{' => {
                let rep = self.parse_brace_repeat()?;
                self.reject_lazy()?;
                Ok(format!("{atom}{rep}"))
            }
            _ => Ok(atom),
        }
    }

    /// Reject lazy quantifier suffix `?` after a quantifier.
    fn reject_lazy(&mut self) -> Result<(), RegexToGbnfError> {
        if self.peek() == Some('?') {
            return Err(RegexToGbnfError::Unsupported(
                "lazy quantifiers (*? +? ?? {n,m}?)".into(),
            ));
        }
        Ok(())
    }

    /// `{n}` / `{n,}` / `{n,m}` — GBNF supports all three natively.
    fn parse_brace_repeat(&mut self) -> Result<String, RegexToGbnfError> {
        // self.pos is at '{'
        self.bump();
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        let min_str: String = self.chars[start..self.pos].iter().collect();
        if min_str.is_empty() {
            return Err(RegexToGbnfError::BadRepeat("{…} needs a number".into()));
        }
        let rep = if self.peek() == Some(',') {
            self.bump();
            let mstart = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
            let max_str: String = self.chars[mstart..self.pos].iter().collect();
            if max_str.is_empty() {
                format!("{{{min_str},}}")
            } else {
                format!("{{{min_str},{max_str}}}")
            }
        } else {
            format!("{{{min_str}}}")
        };
        if self.bump() != Some('}') {
            return Err(RegexToGbnfError::BadRepeat("missing '}'".into()));
        }
        Ok(rep)
    }

    fn parse_atom(&mut self) -> Result<Node, RegexToGbnfError> {
        let Some(c) = self.bump() else {
            return Err(RegexToGbnfError::UnexpectedEnd);
        };
        match c {
            '(' => {
                // Non-capturing group or look-around / named group
                if self.peek() == Some('?') {
                    match self.peek_at(1) {
                        Some(':') => {
                            self.bump();
                            self.bump();
                        }
                        Some('<') => {
                            return Err(RegexToGbnfError::Unsupported(
                                "look-behind / named groups".into(),
                            ));
                        }
                        _ => {
                            return Err(RegexToGbnfError::Unsupported(
                                "look-ahead (?=…)/(?!…)".into(),
                            ));
                        }
                    }
                }
                let inner = self.parse_alt()?;
                if self.bump() != Some(')') {
                    return Err(RegexToGbnfError::UnexpectedEnd);
                }
                // parse_alt already parenthesizes multi-branch alternations —
                // don't double-wrap.
                let gbnf = if inner.starts_with('(') && inner.ends_with(')') {
                    inner
                } else {
                    format!("({inner})")
                };
                Ok(Node { gbnf })
            }
            '[' => {
                let class = self.parse_char_class()?;
                Ok(Node { gbnf: class })
            }
            '.' => Ok(Node {
                gbnf: "[^\\n]".into(),
            }),
            '^' | '$' => {
                // Anchors are implicit — the whole output is the match.
                Ok(Node {
                    gbnf: "\"\"?".into(),
                })
            }
            '\\' => {
                let esc = self.parse_escape()?;
                Ok(Node { gbnf: esc })
            }
            c => Ok(Node {
                gbnf: gbnf_literal(c),
            }),
        }
    }

    fn parse_escape(&mut self) -> Result<String, RegexToGbnfError> {
        let Some(c) = self.bump() else {
            return Err(RegexToGbnfError::UnexpectedEnd);
        };
        let s = match c {
            'd' => "[0-9]".into(),
            'D' => "[^0-9]".into(),
            'w' => "[a-zA-Z0-9_]".into(),
            'W' => "[^a-zA-Z0-9_]".into(),
            's' => "[ \\t\\n\\r]".into(),
            'S' => "[^ \\t\\n\\r]".into(),
            'n' => "\"\\n\"".into(),
            't' => "\"\\t\"".into(),
            'r' => "\"\\r\"".into(),
            '0' => "\"\\x00\"".into(),
            '1'..='9' => {
                return Err(RegexToGbnfError::Unsupported(format!(
                    "backreference \\{c}"
                )));
            }
            // Escaped literal (e.g. `\.`, `\[`, `\\`)
            c => gbnf_literal(c),
        };
        Ok(s)
    }

    /// Parse `[...]` char class including ranges and negation.
    /// GBNF shares the same `[a-zA-Z]`/`[^…]` syntax, so we mostly copy the
    /// class through, translating escaped shorthand inside the class.
    fn parse_char_class(&mut self) -> Result<String, RegexToGbnfError> {
        // self.pos is just after '['
        let mut out = String::from("[");
        if self.peek() == Some('^') {
            self.bump();
            out.push('^');
        }
        // A literal ']' may appear first in the class
        if self.peek() == Some(']') {
            self.bump();
            out.push_str("\\]");
        }
        loop {
            let Some(c) = self.bump() else {
                return Err(RegexToGbnfError::BadCharClass("unclosed '['".into()));
            };
            match c {
                ']' => {
                    out.push(']');
                    return Ok(out);
                }
                '\\' => {
                    let Some(e) = self.bump() else {
                        return Err(RegexToGbnfError::BadCharClass("dangling '\\'".into()));
                    };
                    match e {
                        'd' => out.push_str("0-9"),
                        'w' => out.push_str("a-zA-Z0-9_"),
                        's' => out.push_str(" \\t\\n\\r"),
                        'n' => out.push_str("\\n"),
                        't' => out.push_str("\\t"),
                        'r' => out.push_str("\\r"),
                        'D' | 'W' | 'S' => {
                            return Err(RegexToGbnfError::BadCharClass(
                                "negated shorthand inside class".into(),
                            ));
                        }
                        other => {
                            out.push('\\');
                            out.push(other);
                        }
                    }
                }
                // GBNF requires these to be escaped inside classes
                '[' => out.push_str("\\["),
                other => out.push(other),
            }
        }
    }
}

/// Render a literal character as a GBNF `"…"` string, escaping as needed.
fn gbnf_literal(c: char) -> String {
    match c {
        '"' => "\"\\\"\"".into(),
        '\\' => "\"\\\\\"".into(),
        '\n' => "\"\\n\"".into(),
        '\t' => "\"\\t\"".into(),
        '\r' => "\"\\r\"".into(),
        c if (c as u32) < 0x20 => format!("\"\\x{:02X}\"", c as u32),
        c => format!("\"{c}\""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        assert_eq!(
            regex_to_gbnf("abc").unwrap(),
            "root ::= \"a\" \"b\" \"c\"\n"
        );
    }

    #[test]
    fn test_non_empty_line() {
        // The pattern used by the suggest-title skills: ^.+$
        assert_eq!(
            regex_to_gbnf("^.+$").unwrap(),
            "root ::= \"\"? [^\\n]+ \"\"?\n"
        );
    }

    #[test]
    fn test_char_class() {
        assert_eq!(regex_to_gbnf("[a-z0-9]+").unwrap(), "root ::= [a-z0-9]+\n");
    }

    #[test]
    fn test_negated_class() {
        assert_eq!(regex_to_gbnf("[^abc]*").unwrap(), "root ::= [^abc]*\n");
    }

    #[test]
    fn test_alternation() {
        assert_eq!(
            regex_to_gbnf("cat|dog").unwrap(),
            "root ::= (\"c\" \"a\" \"t\" | \"d\" \"o\" \"g\")\n"
        );
    }

    #[test]
    fn test_group_alternation() {
        assert_eq!(
            regex_to_gbnf("(yes|no)").unwrap(),
            "root ::= (\"y\" \"e\" \"s\" | \"n\" \"o\")\n"
        );
    }

    #[test]
    fn test_shorthand_escapes() {
        assert_eq!(regex_to_gbnf("\\d+").unwrap(), "root ::= [0-9]+\n");
        assert_eq!(regex_to_gbnf("\\w*").unwrap(), "root ::= [a-zA-Z0-9_]*\n");
        assert_eq!(regex_to_gbnf("\\s").unwrap(), "root ::= [ \\t\\n\\r]\n");
    }

    #[test]
    fn test_repeat_braces() {
        assert_eq!(regex_to_gbnf("a{3}").unwrap(), "root ::= \"a\"{3}\n");
        assert_eq!(regex_to_gbnf("a{2,5}").unwrap(), "root ::= \"a\"{2,5}\n");
        assert_eq!(regex_to_gbnf("a{2,}").unwrap(), "root ::= \"a\"{2,}\n");
    }

    #[test]
    fn test_escaped_metachar() {
        assert_eq!(
            regex_to_gbnf("a\\.b").unwrap(),
            "root ::= \"a\" \".\" \"b\"\n"
        );
    }

    #[test]
    fn test_reject_lookahead() {
        assert!(regex_to_gbnf("a(?=b)").is_err());
        assert!(regex_to_gbnf("a(?!b)").is_err());
    }

    #[test]
    fn test_reject_backreference() {
        assert!(regex_to_gbnf("(a)\\1").is_err());
    }

    #[test]
    fn test_reject_lazy() {
        assert!(regex_to_gbnf("a*?").is_err());
        assert!(regex_to_gbnf("a+?").is_err());
    }

    #[test]
    fn test_non_capturing_group() {
        assert_eq!(
            regex_to_gbnf("(?:ab)+").unwrap(),
            "root ::= (\"a\" \"b\")+\n"
        );
    }

    #[test]
    fn test_complex_pattern() {
        // ISO-date-ish: \d{4}-\d{2}-\d{2}
        assert_eq!(
            regex_to_gbnf("\\d{4}-\\d{2}-\\d{2}").unwrap(),
            "root ::= [0-9]{4} \"-\" [0-9]{2} \"-\" [0-9]{2}\n"
        );
    }
}
