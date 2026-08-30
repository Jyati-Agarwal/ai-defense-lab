PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= $(PROJECT_ROOT)/.venv/bin/python
CONFIG ?= $(PROJECT_ROOT)/deploy/apprunner.json
IMAGE ?= ai-defense-lab:app-runner
DEPLOY := $(PYTHON) $(PROJECT_ROOT)/scripts/deploy_apprunner.py --root $(PROJECT_ROOT) --config $(CONFIG) --image $(IMAGE)

.PHONY: help verify release-amended package deploy-plan deploy

help:
	@echo "Use this deployment makefile from any directory:"
	@echo "  make -f $(PROJECT_ROOT)/deploy/Makefile verify"
	@echo "  make -f $(PROJECT_ROOT)/deploy/Makefile release-amended"
	@echo "  make -f $(PROJECT_ROOT)/deploy/Makefile package"
	@echo "  make -f $(PROJECT_ROOT)/deploy/Makefile deploy-plan"
	@echo "  make -f $(PROJECT_ROOT)/deploy/Makefile deploy"

verify:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/release.py --verify

# Explicitly named because the immutable amendment forbids ordinary signing.
# This signs repeatable runtime/package changes; it never runs evaluation.
release-amended:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/release.py --release-policy-amendment
	$(MAKE) -f $(PROJECT_ROOT)/deploy/Makefile verify

package:
	$(DEPLOY) --build-only

deploy-plan:
	$(DEPLOY) --dry-run

deploy:
	$(DEPLOY)
