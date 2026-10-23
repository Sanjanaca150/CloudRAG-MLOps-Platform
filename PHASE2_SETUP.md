# CloudRAG Phase 2 setup

Copy `app/evaluation.py` into the existing project's `app` folder, and copy `evaluation/` and `scripts/` into the project root. Install Phase 2 dependency using the same activated Python 3.12 virtual environment:

```powershell
python -m pip install -r requirements-phase2.txt
```

Start the API in one PowerShell terminal:

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and call `POST /ingest`. Then, in a second PowerShell terminal from the project root with `.venv` activated, run:

```powershell
python scripts/run_evaluation.py
```

Open a third terminal to view experiments:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. The script writes `reports/evaluation_results.json` and logs aggregate metrics and that report as an MLflow artifact.

Metrics are deliberately lightweight and interpretable: expected-source hit rate checks whether retrieval returned the expected document; keyword coverage checks whether configured answer phrases appeared; latency records retrieval + generation time; average cosine distance is a retrieval diagnostic. These are not a full semantic correctness or faithfulness evaluation, and keyword coverage can penalize valid paraphrases.
