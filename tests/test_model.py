import pytest
import torch

from teenie.data import Op, make_dataset, result_bound, split_dataset
from teenie.loop import compute_loss, halt_distribution
from teenie.model import RecursiveArithModel


def test_data_and_split():
    ds = make_dataset(10)
    assert len(ds) == 441
    a, b, op, result = ds.tensors
    assert torch.equal(result, a + b)
    assert (op == 0).all()
    train, val = split_dataset(ds, seed=4)
    assert len(train) == 352 and len(val) == 89
    assert set(train.indices).isdisjoint(val.indices)
    assert train.indices == split_dataset(ds, seed=4)[0].indices


@pytest.mark.parametrize("op", list(Op))
def test_outcome_bounds(op):
    a, b, _, target = make_dataset(3, (op,)).tensors
    expected = {Op.ADD: a + b, Op.SUB: a - b, Op.MUL: a * b}[op]
    assert torch.equal(target, expected)
    assert (target.abs() <= result_bound(3)).all()


def test_forward_and_gradients():
    model = RecursiveArithModel(2, hidden=16, max_steps=4)
    a, b, op, target = make_dataset(2).tensors
    logits, halts = model(a, b, op)
    assert logits.shape == (25, 4, 9)
    assert halts.shape == (25, 4)
    loss = compute_loss(logits, halts, target, model.result_bound)
    loss.backward()
    for layer in (model.cell[0], model.halt_head, model.result_head):
        assert layer.weight.grad.isfinite().all()
        assert layer.weight.grad.abs().sum() > 0


def test_ponder_rewards_early_halting():
    early = torch.full((2, 4), 0.9, requires_grad=True)
    late = torch.full((2, 4), 0.1)
    weights = halt_distribution(early)
    assert torch.allclose(weights.sum(1), torch.ones(2))
    steps = torch.arange(1, 5)
    expected = (weights * steps).sum()
    assert expected < (halt_distribution(late) * steps).sum()
    expected.backward()
    assert (early.grad[:, :-1] < 0).all()


def test_independent_halting(monkeypatch):
    model = RecursiveArithModel(2, hidden=8, max_steps=3)
    sizes = []

    def step(h, ea, eb, t):
        sizes.append(len(h))
        logits = h.new_zeros((len(h), 9))
        logits[:, t] = 1
        halt = h.new_zeros(len(h))
        if t == 0:
            halt[0] = 1
        return h, logits, halt

    monkeypatch.setattr(model, "_step", step)
    x = torch.zeros(2, dtype=torch.long)
    result, steps = model.predict(x, x, x)
    assert steps.tolist() == [1, 3]
    assert result.tolist() == [-4, -2]
    assert sizes == [2, 1, 1]
    _, steps = model.predict(x, x, x, adaptive=False)
    assert steps.tolist() == [3, 3]


def test_single_step_and_validation():
    model = RecursiveArithModel(1, hidden=8, max_steps=1)
    a, b, op, target = make_dataset(1).tensors
    logits, halts = model(a, b, op)
    assert torch.equal(halt_distribution(halts), torch.ones_like(halts))
    assert compute_loss(logits, halts, target, model.result_bound).isfinite()
    assert (model.predict(a, b, op)[1] == 1).all()
    with pytest.raises(ValueError):
        model(a + 10, b, op)
    with pytest.raises(ValueError):
        make_dataset(0)
    with pytest.raises(ValueError):
        make_dataset(1, ())


def test_overfit_ten_examples():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = RecursiveArithModel(2, hidden=32, max_steps=2)
    a, b, op, target = [x[:10] for x in make_dataset(2).tensors]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits, halts = model(a, b, op)
        compute_loss(logits, halts, target, model.result_bound).backward()
        optimizer.step()
    assert torch.equal(model.predict(a, b, op)[0], target)


def test_numeric_loss_and_decode():
    model = RecursiveArithModel(2, hidden=16, representation="numeric")
    a, b, op, target = make_dataset(2, tuple(Op)).tensors
    output, halts = model(a, b, op)
    assert output.shape == (75, 4, 1)
    loss = compute_loss(output, halts, target, model.result_bound, representation="numeric")
    loss.backward()
    for layer in (model.embed_a, model.embed_b, model.cell[0], model.result_head):
        assert layer.weight.grad.isfinite().all()
        assert layer.weight.grad.abs().sum() > 0
    assert model.decode(torch.tensor([[-1.0], [0.0], [0.75], [2.0]])).tolist() == [-4, 0, 3, 4]
    perfect = (target.float() / model.result_bound)[:, None, None].expand(-1, 4, -1)
    assert compute_loss(perfect, halts, target, model.result_bound,
                        ponder_weight=0, representation="numeric").item() == 0


def test_numeric_training_reduces_error():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = RecursiveArithModel(2, hidden=16, max_steps=2, representation="numeric")
    a, b, op, target = make_dataset(2, tuple(Op)).tensors
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def loss():
        output, halt = model(a, b, op)
        return compute_loss(output[:, -1:], halt[:, -1:], target, model.result_bound,
                            ponder_weight=0, representation="numeric")

    initial = loss().item()
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        error = loss()
        error.backward()
        optimizer.step()
    assert loss().item() < initial / 20
