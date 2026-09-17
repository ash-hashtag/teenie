import argparse
import json

import pytest
import torch

from teenie.checkpoint import load_checkpoint, save_checkpoint
from teenie.data import Op, make_dataset
from teenie.model import RecursiveArithModel
from teenie.train import evaluate, main, parse_ops, select_device


@pytest.mark.parametrize("representation", ["embedding", "numeric"])
def test_checkpoint_roundtrip(tmp_path, representation):
    model = RecursiveArithModel(2, hidden=16, max_steps=3, representation=representation)
    a, b, op, _ = make_dataset(2).tensors
    before = model.predict(a, b, op)
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(path, model, ops=(Op.ADD, Op.MUL), adaptive=False, epoch=7, seed=5)
    restored, info = load_checkpoint(path)
    assert info["ops"] == [0, 2]
    assert info["adaptive"] is False
    assert info["epoch"] == 7 and info["seed"] == 5
    assert not restored.training
    assert restored.representation == representation
    assert all(torch.equal(x, y) for x, y in zip(before, restored.predict(a, b, op)))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_ops_and_device(monkeypatch):
    assert parse_ops("add,sub,mul") == (Op.ADD, Op.SUB, Op.MUL)
    assert parse_ops("0, 2") == (Op.ADD, Op.MUL)
    for value in ("", "div", "add,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ops(value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="CUDA requested"):
        select_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("auto").type == "cuda"


def test_train_load_eval_and_finetune(tmp_path, monkeypatch, capsys):
    path = tmp_path / "first.pt"
    monkeypatch.setattr("sys.argv", ["teenie", "--device", "cpu", "--epochs", "1",
                                    "--num-range", "2", "--hidden", "8", "--ops", "add,sub",
                                    "--no-ponder", "--save", str(path)])
    main()
    trained = json.loads(capsys.readouterr().out)
    assert "all_sub" in trained and "zero_shot_sub" not in trained
    assert "zero_shot_mul" in trained
    assert trained["train"]["steps"] == 4
    original = path.read_bytes()
    assert (tmp_path / "first.best.pt").exists()
    monkeypatch.setattr("sys.argv", ["teenie", "--device", "cpu", "--load", str(path), "--eval-only"])
    main()
    assert json.loads(capsys.readouterr().out) == trained
    assert path.read_bytes() == original
    second = tmp_path / "second.pt"
    monkeypatch.setattr("sys.argv", ["teenie", "--device", "cpu", "--load", str(path),
                                    "--epochs", "1", "--ops", "add,sub,mul", "--save", str(second)])
    main()
    assert "zero_shot_mul" not in json.loads(capsys.readouterr().out)
    assert load_checkpoint(second)[1]["ops"] == [0, 1, 2]
    monkeypatch.setattr("sys.argv", ["teenie", "--device", "cpu", "--load", str(second),
                                    "--epochs", "1", "--ops", "add", "--save", str(second)])
    main()
    metrics = json.loads(capsys.readouterr().out)
    assert "previously_trained_sub" in metrics and "previously_trained_mul" in metrics
    assert not any(key.startswith("zero_shot") for key in metrics)
    assert load_checkpoint(second)[1]["seen_ops"] == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("representation", ["embedding", "numeric"])
def test_cuda_train_save_load(tmp_path, representation):
    from teenie.loop import compute_loss

    model = RecursiveArithModel(2, hidden=8, representation=representation).to("cuda")
    a, b, op, target = (x.to("cuda") for x in make_dataset(2, tuple(Op)).tensors)
    optimizer = torch.optim.AdamW(model.parameters())
    logits, halts = model(a, b, op)
    compute_loss(logits, halts, target, model.result_bound, representation=representation).backward()
    optimizer.step()
    expected = model.predict(a, b, op)
    path = tmp_path / "cuda.pt"
    save_checkpoint(path, model, ops=tuple(Op), adaptive=True, epoch=1, seed=0)
    restored, _ = load_checkpoint(path, "cuda")
    assert all(torch.equal(x, y) for x, y in zip(expected, restored.predict(a, b, op)))
    cpu_model, _ = load_checkpoint(path, "cpu")
    assert next(cpu_model.parameters()).device.type == "cpu"


def test_legacy_checkpoint(tmp_path):
    model = RecursiveArithModel(2, hidden=8)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, ops=(Op.ADD,), adaptive=True, epoch=1, seed=0)
    content = torch.load(path, weights_only=True)
    del content["model_config"]["representation"]
    torch.save(content, path)
    restored, _ = load_checkpoint(path)
    assert restored.representation == "embedding"
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_best_checkpoint_selection(tmp_path, monkeypatch, capsys):
    # Reports evaluate five loaders each: train, val, and one grid per operation.
    calls = 0

    def fake_evaluate(*args):
        nonlocal calls
        epoch_index = calls // 5
        calls += 1
        return {"accuracy": [0.5, 0.9, 0.7][epoch_index], "steps": 4.0}

    monkeypatch.setattr("teenie.train.evaluate", fake_evaluate)
    path = tmp_path / "numeric.pt"
    monkeypatch.setattr("sys.argv", ["teenie", "--device", "cpu", "--epochs", "3",
                                    "--log-every", "1", "--representation", "numeric",
                                    "--num-range", "2", "--hidden", "8", "--no-ponder",
                                    "--lr-schedule", "cosine", "--save", str(path)])
    main()
    assert load_checkpoint(path)[1]["epoch"] == 3
    assert load_checkpoint(tmp_path / "numeric.best.pt")[1]["epoch"] == 2


def test_per_operation_metrics(monkeypatch):
    from torch.utils.data import DataLoader

    model = RecursiveArithModel(1, hidden=8)
    monkeypatch.setattr(model, "predict", lambda a, b, op, adaptive:
                        (torch.zeros_like(a), torch.ones_like(a)))
    metrics = evaluate(model, DataLoader(make_dataset(1, tuple(Op)), batch_size=4))
    assert metrics["accuracy"] == 11 / 27
    assert metrics["per_op"] == {
        "add": {"accuracy": 3 / 9, "count": 9},
        "sub": {"accuracy": 3 / 9, "count": 9},
        "mul": {"accuracy": 5 / 9, "count": 9},
    }
