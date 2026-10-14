import boto3
from moto import mock_aws
from peftkit.publish import from_s3, to_local, to_s3


def _artifact(tmp_path):
    src = tmp_path / "adapter"
    (src / "sub").mkdir(parents=True)
    (src / "adapter_config.json").write_text("{}")
    (src / "sub" / "w.safetensors").write_bytes(b"123")
    (src / "optimizer.pt").write_bytes(b"skip me")
    return src


def test_local_copy_skips_optimizer(tmp_path):
    dst = to_local(str(_artifact(tmp_path)), str(tmp_path / "drive" / "run1"))
    assert (tmp_path / "drive/run1/sub/w.safetensors").exists()
    assert not (tmp_path / "drive/run1/optimizer.pt").exists() and dst


@mock_aws
def test_s3_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    boto3.client("s3").create_bucket(Bucket="models")
    keys = to_s3(str(_artifact(tmp_path)), "s3://models/runs/r1")
    assert sorted(keys) == ["runs/r1/adapter_config.json", "runs/r1/sub/w.safetensors"]
    assert from_s3("s3://models/runs/r1", str(tmp_path / "back")) == 2
    assert (tmp_path / "back/sub/w.safetensors").read_bytes() == b"123"
