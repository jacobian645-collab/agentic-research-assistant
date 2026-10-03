# Offline web fixture corpus

`web_corpus.json` backs `WEB_SEARCH_PROVIDER=offline`. It contains short, **paraphrased summaries**
of public sources (papers, vendor docs) keyed by their real URLs, so the full agent graph can run
deterministically in tests, CI and demos without network access or API keys.

* Summaries are not verbatim quotes — always consult the linked original for authoritative wording.
* Pricing / availability facts reflect public documentation as of October 2026 and will drift.
* `agent-tricks.example.com` is a deliberately malicious page used to exercise the
  prompt-injection tripwire (`guardrails.scan_for_injection`).
