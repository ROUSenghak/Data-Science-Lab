.PHONY: setup api web dev test fixtures

setup:            ## install Python (venv) and Node dependencies
	cd api && python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
	cd web && npm install

api:              ## Python execution service on :8000
	cd api && .venv/bin/uvicorn app.main:app --port 8000

web:              ## Next.js dev server on :3000 (proxies /api/py -> :8000)
	cd web && npm run dev

dev:              ## both servers
	$(MAKE) -j2 api web

test:             ## all numerical tests + type check + lint
	cd api && .venv/bin/python -m pytest -q
	cd web && npm test && npm run typecheck && npm run lint

fixtures:         ## regenerate the NumPy reference trajectory used by the TypeScript tests
	cd api && .venv/bin/python -m scripts.export_fixtures
