.PHONY: help clean build check publish publish-test test

help:
	@echo "clean         remove dist/ so stale artifacts can never be published"
	@echo "build         clean, then build sdist + wheel"
	@echo "check         build, then validate metadata with twine"
	@echo "publish       check, then upload to PyPI"
	@echo "publish-test  check, then upload to TestPyPI"
	@echo "test          run the test suite"
	@echo ""
	@echo "Credentials come from the environment, never this file:"
	@echo "  UV_PUBLISH_TOKEN=pypi-... make publish"

clean:
	rm -rf dist build src/*.egg-info

# clean is a prerequisite, not a follow-up: dist/ must be empty before we
# build, or `uv publish dist/*` would pick up artifacts from an older version.
build: clean
	uv build

check: build
	uvx twine check dist/*

publish: check
	@test -n "$$UV_PUBLISH_TOKEN" || { \
		echo "UV_PUBLISH_TOKEN is not set."; \
		exit 1; \
	}
	uv publish dist/*

test:
	uv run pytest
