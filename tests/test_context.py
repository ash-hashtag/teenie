from dataclasses import replace
import json

import pytest
import torch

from teenie.context_checkpoint import load_context_checkpoint, save_context_checkpoint
from teenie.context_data import KnownEpisodeSampler, known_results, pair_split
from teenie.context_eval import ablate, division_episodes, evaluate_context
from teenie.context_model import ContextArithModel
from teenie.context_predict import predict_from_context


def test_training_has_no_division_or_query_leakage():
    sampler = KnownEpisodeSampler()
    batch = sampler.batch(64)
    assert batch.support_pairs.shape == (64, 99, 2)
    assert batch.support_ops.shape == (64, 99, 4)
    assert (batch.support_ops[..., 3] == 0).all()
    assert (batch.query_ops[..., 3] == 0).all()
    assert not (batch.support_pairs == batch.query_pairs[:, None]).all(-1).any()
    for support in batch.support_pairs:
        assert len(support.unique(dim=0)) == 99
    with pytest.raises(ValueError, match="only support"):
        known_results(torch.tensor([[7, 2]]), torch.tensor([3]))
    # Every episode corresponds to one of the three permitted functions,
    # despite randomizing the one-hot operation identity.
    for i in range(len(batch.targets)):
        assert any(torch.equal(known_results(batch.support_pairs[i], torch.tensor(op)), batch.support_results[i])
                   and known_results(batch.query_pairs[i], torch.tensor(op)) == batch.targets[i]
                   for op in range(3))


def test_validation_support_only_uses_training_pairs():
    sampler = KnownEpisodeSampler()
    val = sampler.validation()
    pairs, tr, va = pair_split()
    training_pairs = set(map(tuple, pairs[tr].tolist()))
    validation_pairs = set(map(tuple, pairs[va].tolist()))
    assert set(map(tuple, val.support_pairs.flatten(0, 1).tolist())) <= training_pairs
    assert set(map(tuple, val.query_pairs.tolist())) == validation_pairs
    assert (val.query_ops[:, 3] == 0).all()
    same = KnownEpisodeSampler().validation()
    assert torch.equal(val.support_pairs, same.support_pairs)


def test_division_context_excludes_query_and_zero_divisors():
    episodes = division_episodes()
    assert episodes.query_pairs.shape == (420, 2)
    assert episodes.support_pairs.shape == (420, 99, 2)
    assert (episodes.support_ops[..., 3] == 1).all()
    assert (episodes.query_ops[:, 3] == 1).all()
    assert (episodes.support_pairs[..., 1] != 0).all()
    assert not (episodes.support_pairs == episodes.query_pairs[:, None]).all(-1).any()
    for pairs, results in zip(episodes.support_pairs, episodes.support_results):
        assert len(pairs.unique(dim=0)) == 99
        assert torch.equal(results, torch.div(pairs[:, 0], pairs[:, 1], rounding_mode="floor"))
    assert division_episodes(query=[-7, 2]).targets.item() == -4
    assert division_episodes(query=[7, -2]).targets.item() == -4
    with pytest.raises(ValueError):
        division_episodes(query=[1, 0])
    with pytest.raises(ValueError):
        division_episodes(num_range=1, context_size=99)


def test_context_model_invariants_and_gradients():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = ContextArithModel(hidden=16, heads=2, steps=2)
    batch = KnownEpisodeSampler().batch(3)
    output = model(*batch.inputs())
    assert output.shape == (3,)
    # The query answer is not a model input.
    wrong_targets = replace(batch, targets=batch.targets + 999)
    assert torch.equal(output, model(*wrong_targets.inputs()))
    # Applying a new label consistently uses exactly the same computation.
    fourth_support = torch.zeros_like(batch.support_ops)
    fourth_support[..., 3] = 1
    fourth_query = torch.zeros_like(batch.query_ops)
    fourth_query[..., 3] = 1
    new_op = replace(batch, support_ops=fourth_support, query_ops=fourth_query)
    assert torch.equal(output, model(*new_op.inputs()))
    order = torch.randperm(99)
    reordered = replace(batch, support_pairs=batch.support_pairs[:, order],
                        support_ops=batch.support_ops[:, order], support_results=batch.support_results[:, order])
    assert torch.allclose(output, model(*reordered.inputs()), atol=1e-6)
    assert not torch.allclose(output, model(*ablate(batch, "zero_results").inputs()))
    assert model(*ablate(batch, "no_context").inputs()).isfinite().all()
    results = batch.support_results.float().requires_grad_(True)
    model(*replace(batch, support_results=results).inputs()).sum().backward()
    assert results.grad.abs().sum() > 0
    assert model.attention.in_proj_weight.grad.isfinite().all()
    # A context with no matching operator must not silently produce NaNs.
    with pytest.raises(ValueError, match="same operation"):
        model(*replace(batch, query_ops=fourth_query).inputs())


def test_ablation_and_evaluation_do_not_update_model():
    torch.set_num_threads(1)
    model = ContextArithModel(hidden=8, heads=2, steps=1)
    batch = division_episodes(query=[7, 2])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    shuffled = ablate(batch, "shuffled_results")
    assert torch.equal(shuffled.support_pairs, batch.support_pairs)
    assert torch.equal(shuffled.support_results.sort().values, batch.support_results.sort().values)
    assert torch.equal(batch.support_results, division_episodes(query=[7, 2]).support_results)
    report = evaluate_context(model, batch)
    assert set(report) == {"correct", "shuffled_results", "zero_results", "no_context"}
    assert all(value["count"] == 1 for value in report.values())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_custom_context_requires_no_reference_operation():
    model = ContextArithModel(hidden=8, heads=2, steps=1)
    # These are arbitrary user-provided labels, not calculated by the predictor.
    document = {"examples": [{"a": 1, "b": 2, "op": [0, 0, 0, 1], "result": 5},
                              {"a": 3, "b": 4, "op": [0, 0, 0, 1], "result": -2}],
                "query": {"a": 7, "b": 8, "op": [0, 0, 0, 1]}}
    result = predict_from_context(model, document)
    assert isinstance(result["prediction"], int)
    assert result["context_size"] == 2
    document["query"]["result"] = 123
    with pytest.raises(ValueError, match="answer"):
        predict_from_context(model, document)
    document["query"] = {"a": 1, "b": 2, "op": [0, 0, 0, 1]}
    with pytest.raises(ValueError, match="demonstrations"):
        predict_from_context(model, document)


def test_context_cli_train_save_load_eval(tmp_path, monkeypatch, capsys):
    from teenie.context_train import main as train
    from teenie.context_eval import main as evaluate

    path = tmp_path / "context.pt"
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--num-range", "2",
                                    "--context-size", "5", "--hidden", "8", "--heads", "2",
                                    "--updates", "2", "--batch-size", "4", "--save", str(path)])
    train()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert set(records[-1]["val_per_op"]) == {"add", "sub", "mul"}
    model, metadata = load_context_checkpoint(tmp_path / "context.best.pt")
    assert metadata["training_ops"] == [0, 1, 2]
    assert not model.training
    report_path = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["eval", "--device", "cpu", "--load", str(path),
                                    "--context-size", "5", "--query", "-1", "2",
                                    "--output", str(report_path)])
    original = path.read_bytes()
    evaluate()
    result = json.loads(capsys.readouterr().out)
    assert result["predictions"][0]["target"] == -1
    assert result["division_trained"] is False
    assert len(result["demonstrations"]) == 5
    assert json.loads(report_path.read_text()) == result
    assert path.read_bytes() == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_context_cuda_roundtrip(tmp_path):
    model = ContextArithModel(hidden=16, heads=2, steps=2).to("cuda")
    batch = KnownEpisodeSampler().batch(4).to("cuda")
    opt = torch.optim.AdamW(model.parameters())
    (model(*batch.inputs()) - batch.targets / model.result_bound).square().mean().backward()
    opt.step()
    path = tmp_path / "gpu.pt"
    save_context_checkpoint(path, model, {"seed": 0, "context_size": 99}, 1, {})
    restored, _ = load_context_checkpoint(path, "cuda")
    assert torch.equal(model.predict(*batch.inputs()), restored.predict(*batch.inputs()))
    cpu, _ = load_context_checkpoint(path)
    assert next(cpu.parameters()).device.type == "cpu"
