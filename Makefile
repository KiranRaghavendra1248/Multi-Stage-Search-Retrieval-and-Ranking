-include .env
export

REMOTE := $(VAST_AI_USER)@$(VAST_AI_IP)
SSH_PORT := $(or $(VAST_AI_PORT),22)

.PHONY: venv setup-local setup-remote test sync-push sync-pull-triplets sync-pull-model sync-pull-bm25 sync-pull-results phase1 phase2 phase3 phase4 phase5 phase6 run-all start-vllm start-vllm-teacher stop-vllm-teacher

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

phase2:
	mkdir -p logs
	.venv/bin/python scripts/phase2_mine_negatives.py 2>&1 | tee logs/phase2.log

phase3:
	mkdir -p logs
	.venv/bin/python scripts/phase3_train_biencoder.py 2>&1 | tee logs/phase3.log

phase4:
	mkdir -p logs
	.venv/bin/python scripts/phase4_build_index.py 2>&1 | tee logs/phase4.log

phase5:
	mkdir -p logs
	.venv/bin/python scripts/phase5_inference_demo.py --pipeline A 2>&1 | tee logs/phase5.log

phase6:
	mkdir -p logs
	.venv/bin/python scripts/phase6_evaluate.py 2>&1 | tee logs/phase6.log

run-all: phase2 phase3 phase4 phase6
	@echo "All phases complete. Check logs/ for output."

sync-push:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" --exclude='/data/' --exclude='__pycache__' --exclude='.venv/' \
		./ $(REMOTE):/workspace/retrieval/

start-vllm:
	.venv/bin/python -m vllm.entrypoints.openai.api_server \
		--model $(VLLM_MODEL) \
		--quantization awq \
		--gpu-memory-utilization 0.6 \
		--max-model-len 4096 \
		--port 8000 &
	@echo "vLLM HyDE server starting on port 8000. Wait ~60s before running inference."

# Dense teacher embedding server for Phase 2 mining (e5-mistral-7b-instruct).
# Uses INT8 quantization (~7-8 GB VRAM) leaving room for the FAISS index.
# NOTE: Cannot run alongside start-vllm (16 GB VRAM constraint).
#       Stop this server before running Phase 5/6 inference.
start-vllm-teacher:
	.venv/bin/python -m vllm.entrypoints.openai.api_server \
		--model $(TEACHER_MODEL) \
		--task embedding \
		--quantization bitsandbytes \
		--load-format bitsandbytes \
		--port 8001 &
	@echo "vLLM teacher embedding server starting on port 8001. Wait ~60s before running phase2."

stop-vllm-teacher:
	@pkill -f "port 8001" || true
	@echo "vLLM teacher server (port 8001) stopped."

sync-pull-triplets:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/data/triplets/ \
		./data/triplets/

sync-pull-model:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/data/checkpoints/best_model/ \
		./data/checkpoints/best_model/

sync-pull-bm25:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/data/index/bm25/ \
		./data/index/bm25/

sync-pull-results:
	rsync -avz --progress -e "ssh -p $(SSH_PORT) -i ~/.ssh/id_vastai" \
		$(REMOTE):/workspace/retrieval/results/ \
		./results/
