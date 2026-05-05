PYTHON ?= python3

.PHONY: run validate clean

run:
	$(PYTHON) pipeline.py

validate:
	$(PYTHON) validate.py

clean:
	rm -f data_manifest.json metrics.json critiques.json report.md walk_forward.json parameter_sensitivity.json adversarial_scenarios.json comparative_brief.md llm_calls.jsonl pipeline_state.json
	rm -f specs/*.json ledgers/*.csv
