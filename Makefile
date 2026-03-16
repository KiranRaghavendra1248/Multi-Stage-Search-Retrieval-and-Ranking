BASE_DIR := $(shell pwd)
REMOTE := $(VAST_AI_USER)@$(VAST_AI_IP)

.PHONY: setup-local setup-remote test sync-push sync-pull-model phase1

setup-local:
	pip install -r requirements/local.txt
	pip install -e .
	python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
	@echo "Note: install faiss-cpu via conda: conda install -c conda-forge faiss-cpu"

setup-remote:
	pip install -r requirements/remote.txt
	pip install -e .
	python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

test:
	python -m pytest tests/ -v

phase1:
	python scripts/phase1_local_dev.py

sync-push:
	rsync -avz --progress --exclude='data/' --exclude='__pycache__' \
		$(BASE_DIR)/ $(REMOTE):/workspace/retrieval/

sync-pull-model:
	rsync -avz --progress \
		$(REMOTE):/workspace/retrieval/data/checkpoints/best_model/ \
		$(BASE_DIR)/data/checkpoints/best_model/
