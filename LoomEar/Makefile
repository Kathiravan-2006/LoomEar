# LoomEar Automation Makefile

.PHONY: setup data features train evaluate audit test app edge clean

setup:
	py -m pip install -r requirements.txt

data:
	py scripts/generate_synthetic_dataset.py

features:
	py scripts/run_eda.py

train:
	py scripts/train_and_evaluate.py

evaluate:
	py scripts/train_and_evaluate.py

audit:
	py scripts/audit_leakage.py

test:
	py -m pytest tests -v

app:
	py -m streamlit run app/dashboard.py

edge:
	py edge/benchmark.py
	py edge/record_and_score.py --max-cycles 5

clean:
	rm -rf __pycache__ .pytest_cache
