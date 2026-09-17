from dataclasses import replace
import json

import pytest
import torch

from teenie.relation_data import RelationSampler, known_values
from teenie.relation_eval import division_contexts, context_variant
from teenie.relation_model import RelationModel, load_relation, predict_document, save_relation


def test_anonymous_training_context_excludes_queries():
    sampler = RelationSampler()
    batch = sampler.batch(8)
    assert len(batch.inputs()) == 3  # context x/y and query x; no target or op ID
    assert batch.context_x.shape == (8, 99, 2)
    assert batch.context_x.min() >= 0 and batch.context_x.max() <= 20
    assert not (batch.context_x == batch.query_x[:, None]).all(-1).any()
    for x in batch.context_x:
        assert len(x.unique(dim=0)) == 99
    with pytest.raises(ValueError, match="restricted"):
        known_values(torch.tensor([[7, 2]]), torch.tensor([3]))
    val = sampler.validation()
    pool = set(map(tuple, sampler.pairs[sampler.train].tolist()))
    assert set(map(tuple, val.context_x.flatten(0, 1).tolist())) <= pool
    assert not set(map(tuple, val.query_x.tolist())) & pool


def test_full_context_division_has_exactly_one_missing_pair():
    batch = division_contexts(context_size=419)
    assert len(batch.targets) == 420
    assert not (batch.context_x == batch.query_x[:, None]).all(-1).any()
    assert (batch.context_x[..., 1] > 0).all()
    for x in batch.context_x:
        assert len(x.unique(dim=0)) == 419
    assert torch.equal(batch.context_y, torch.div(batch.context_x[..., 0], batch.context_x[..., 1], rounding_mode="floor"))
    with pytest.raises(ValueError):
        division_contexts(context_size=420)


def test_relation_is_continuous_context_regression():
    torch.set_num_threads(1)
    model = RelationModel(5)
    batch = RelationSampler(5, 12).batch(3)
    results = batch.context_y.double().requires_grad_(True)
    output = model(batch.context_x, results, batch.query_x)
    assert output.shape == (3,) and output.isfinite().all()
    output.sum().backward()
    assert results.grad.abs().sum() > 0
    assert model.log_length.grad.isfinite().all() and model.log_length.grad.abs().sum() > 0
    assert model.log_ridge.grad.isfinite()
    # Context values are continuous observations, not an operation classifier.
    shifted = model(batch.context_x, results.detach() + 1000, batch.query_x)
    assert torch.allclose(shifted, output.detach() + 1000, atol=1e-7)
    assert (shifted > 900).all()  # not clipped to a fixed result vocabulary
    order = torch.randperm(12)
    assert torch.allclose(output, model(batch.context_x[:, order], results[:, order], batch.query_x), atol=1e-7)
    # A hidden answer cannot affect model inputs.
    changed = replace(batch, targets=batch.targets + 10000)
    assert torch.equal(model(*batch.inputs()), model(*changed.inputs()))
    assert torch.equal(model(*context_variant(batch, "none", 0).inputs()), torch.zeros(3, dtype=torch.float64))


def test_relation_document_and_freeze():
    model = RelationModel()
    document = {"examples": [{"a": 0, "b": 0, "c": 2000.25},
                              {"a": 1, "b": 2, "c": 2000.25}],
                "query": {"a": 3, "b": 4}}
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = predict_document(model, document)
    assert result["prediction"] == 2000.25
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in model.parameters())
    document["query"]["c"] = 0
    with pytest.raises(ValueError, match="WITHOUT"):
        predict_document(model, document)
    del document["query"]["c"]
    document["query"]["op"] = [0, 0, 0, 1]
    with pytest.raises(ValueError, match="operation IDs"):
        predict_document(model, document)


def test_relation_train_save_evaluate_cli(tmp_path, monkeypatch, capsys):
    from teenie.relation_train import main as train
    from teenie.relation_eval import main as evaluate

    path = tmp_path / "relation.pt"
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--operand-max", "3",
                                    "--context-size", "5", "--updates", "2", "--batch-size", "2",
                                    "--save", str(path)])
    train()
    assert len(capsys.readouterr().out.splitlines()) == 2
    model, metadata = load_relation(tmp_path / "relation.best.pt")
    assert metadata["training_relations"] == ["add", "sub", "mul"]
    assert not model.training
    original = path.read_bytes()
    report_path = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["eval", "--device", "cpu", "--load", str(path),
                                    "--contexts", "5", "11", "--output", str(report_path)])
    evaluate()
    result = json.loads(capsys.readouterr().out)
    assert result["operation_menu"] is False
    assert result["division_trained"] is False
    assert result["division"]["11"]["correct"]["count"] == 12
    assert path.read_bytes() == original
    assert json.loads(report_path.read_text()) == result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_relation_cuda_gradient_checkpoint(tmp_path):
    model = RelationModel(5).to("cuda")
    batch = RelationSampler(5, 12).batch(4).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    (model(*batch.inputs()) - batch.targets).square().mean().backward()
    optimizer.step()
    path = tmp_path / "cuda.pt"
    save_relation(path, model, {})
    restored, _ = load_relation(path, "cuda")
    assert torch.equal(model(*batch.inputs()), restored(*batch.inputs()))
    cpu, _ = load_relation(path)
    assert torch.allclose(restored(*batch.inputs()).cpu(), cpu(*batch.to("cpu").inputs()), atol=1e-6)
