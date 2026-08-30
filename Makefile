PYTHON ?= .venv/bin/python
UVICORN ?= .venv/bin/uvicorn
ARENA_FLAGS ?=
MODEL_ID = sentence-transformers/all-MiniLM-L6-v2
MODEL_CACHE := $(CURDIR)/data/models/huggingface
MODEL_REPO_CACHE := $(MODEL_CACHE)/hub/models--sentence-transformers--all-MiniLM-L6-v2
IMMUTABLE_RESULT_PATTERNS = entity_fidelity*.json completejourney_source_*.json shuffle_control_protocol*.json shuffle_null_diagnosis*.json agentdojo_banking_official.json evaluation_*preregistration.json evaluation_*arena_stop_fail.json evaluation_attempt*.json evaluation_release.json evaluation_v*_release.json
export FAISS_OPT_LEVEL ?= generic
export HF_HOME := $(MODEL_CACHE)
export HF_HUB_OFFLINE ?= 1
export TRANSFORMERS_OFFLINE ?= 1

.PHONY: setup model-verify reference reference-uci352 reference-uci502 atlas fidelity entity-freeze entity-exploratory entity-readiness entity-sealed entity-verify evaluation-preregister sim llm-cache train arena arena-rerun eval vaccinate external frontend release release-verify test e2e design-jury-populate design-jury-from-cache container-smoke video acceptance all app clean

setup:
	uv python install 3.11
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch>=2.7,<3'
	uv pip install --python .venv/bin/python -r requirements.lock
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$(MODEL_ID)', device='cpu')"
	cd app/frontend && npm ci
	cd app/frontend && npx playwright install chromium

model-verify:
	@test -d "$(MODEL_REPO_CACHE)" || { echo "local MiniLM cache is missing; run make setup" >&2; exit 2; }
	$(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$(MODEL_ID)', device='cpu', local_files_only=True)"

# The normal pipeline needs only the public marginal reference. Entity-fidelity
# workbooks stay behind explicit dataset-named targets and are never opened by `all`.
reference:
	$(PYTHON) scripts/download_reference.py

reference-uci352:
	$(PYTHON) scripts/download_retail_reference.py

reference-uci502:
	$(PYTHON) scripts/download_retail_ii_reference.py

atlas:
	$(PYTHON) -m atlas.render

fidelity:
	$(PYTHON) -m sim.fidelity --reference-csv data/reference/creditcard_marginals.csv --output results/fidelity.json --gate-200k

entity-freeze:
	$(PYTHON) -m sim.entity_fidelity_v4 freeze

entity-exploratory:
	$(PYTHON) -m sim.entity_fidelity_v4 exploratory

entity-readiness:
	$(PYTHON) -m sim.entity_fidelity_v4 readiness

entity-sealed:
	@test -n "$(CONFIRMATION)" || { echo "CONFIRMATION=APPROVED_SINGLE_FINAL_SCORE:<protocol_sha256> is required" >&2; exit 2; }
	$(PYTHON) -m sim.entity_fidelity_v4 sealed --confirmation "$(CONFIRMATION)"

entity-verify:
	$(PYTHON) -m sim.entity_fidelity_v4 verify

evaluation-preregister:
	$(PYTHON) scripts/release.py --preregister-evaluation

sim:
	$(PYTHON) -m sim.population --events 50000 --seed 20260829 --evaluation-preregistration results/evaluation_v4_preregistration.json --output data/events.jsonl

llm-cache:
	$(PYTHON) -m sim.content_llm --populate-openai

train:
	$(PYTHON) -m defend.train

arena:
	$(PYTHON) -m loop.arena $(ARENA_FLAGS)

arena-rerun:
	$(MAKE) arena ARENA_FLAGS=--rerun

eval:
	$(PYTHON) -m defend.evaluate

vaccinate:
	$(PYTHON) -m defend.vaccination

external:
	$(PYTHON) -m sim.external_challenge --output results/external_attack_challenge.json

frontend:
	cd app/frontend && npm run build

release:
	$(PYTHON) scripts/release.py
	$(PYTHON) scripts/release.py --verify

release-verify:
	$(PYTHON) scripts/release.py --verify

test:
	$(PYTHON) -m pytest -q
	cd app/frontend && npm test

e2e:
	cd app/frontend && npm run test:e2e:real

design-jury-populate:
	$(PYTHON) scripts/design_jury.py --populate-openai

design-jury-from-cache:
	$(PYTHON) scripts/design_jury.py --from-cache

container-smoke:
	./scripts/container_smoke.sh

video:
	./scripts/record_demo_video.sh

# One recipe keeps `make -j all` ordered. The sealed partition is never opened here.
all:
	$(MAKE) model-verify
	$(MAKE) atlas
	$(MAKE) reference
	$(MAKE) fidelity
	$(MAKE) sim
	$(MAKE) train
	$(MAKE) arena-rerun
	$(MAKE) eval
	$(MAKE) vaccinate
	$(MAKE) external
	$(MAKE) frontend
	$(MAKE) release
	$(MAKE) test
	$(MAKE) e2e
	$(MAKE) release-verify

acceptance:
	$(MAKE) release-verify
	$(MAKE) e2e
	$(MAKE) design-jury-from-cache
	$(MAKE) container-smoke
	$(MAKE) video
	$(MAKE) release-verify

app:
	$(UVICORN) app.backend.main:app --host 0.0.0.0 --port 8000

clean:
	$(PYTHON) -c "from fnmatch import fnmatch; from pathlib import Path; patterns='$(IMMUTABLE_RESULT_PATTERNS)'.split(); [p.unlink() for p in Path('results').glob('*') if p.is_file() and p.stat().st_mode & 0o200 and not any(fnmatch(p.name, pattern) for pattern in patterns)]"
