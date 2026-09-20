from kaggle_vllm_nebius.evidence.loader import load_evidence
from kaggle_vllm_nebius.storage import SQLiteStore


def test_sqlite_store_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "runs.db")
    bundle = load_evidence("artifacts/public/normalized/qwen-tp1-c08.json")
    store.put_run(bundle)

    loaded = store.get_run(bundle.run_id)
    assert loaded == bundle
    assert store.list_run_ids() == [bundle.run_id]
