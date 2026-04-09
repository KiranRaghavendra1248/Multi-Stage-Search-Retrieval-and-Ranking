-include .env
export

REMOTE := $(VAST_AI_USER)@$(VAST_AI_IP)
SSH_PORT := $(or $(VAST_AI_PORT),22)

.PHONY: venv setup-local setup-remote test sync-push sync-pull-triplets sync-pull-model phase1

venv:
	python3 -m venv .venv
	@echo "Venv created. Activate with:"
	@echo "  source .venv/bin/activate"

setup-local: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements/local.txt
	.venv/bin/pip install -e .
	.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"

setup-remote:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements/remote.txt
	.venv/bin/pip install -e .
	.venv/bin/python -c "import nltk; [nltk.download(r, download_dir='/root/nltk_data') for r in ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']]"

test:
	.venv/bin/python -m pytest tests/ -v

phase1:
	.venv/bin/python scripts/phase1_local_dev.py

sync-push:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" --exclude='/data/' --exclude='__pycache__' --exclude='.venv/' \
		./ $(REMOTE):/workspace/retrieval/

sync-pull-triplets:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/data/triplets/ \
		./data/triplets/

sync-pull-model:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/data/checkpoints/best_model/ \
		./data/checkpoints/best_model/
