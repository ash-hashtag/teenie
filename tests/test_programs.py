import json

import pytest
import torch

from teenie.program_executor import Program, ProgramInducer, enumerate_programs, execute_programs, trace_program
from teenie.program_model import ArithmeticPrimitives, load_primitives, primitive_transitions, save_primitives
from teenie.program_eval import task_episodes


def oracle_primitives(upper=5, device="cpu"):
    """Only VM unit tests supply exact weights; experiment weights are trained."""
    model = ArithmeticPrimitives(upper).to(device)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1., 1.], [1., -1.]], device=device))
        model.linear.bias.zero_()
    return model


def test_training_data_and_gradients():
    rows = primitive_transitions(5)
    assert set(rows[:, 2].tolist()) == {0, 1}
    assert len(rows.unique(dim=0)) == len(rows)
    assert (rows[:, 0] > 5).any()  # actual multiplication intermediate states
    expected = torch.where(rows[:, 2] == 0, rows[:, 0] + rows[:, 1], rows[:, 0] - rows[:, 1])
    assert torch.equal(rows[:, 3], expected)
    torch.manual_seed(0)
    model = ArithmeticPrimitives(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    def loss():
        return (model(*rows[:, :3].unbind(1)) - rows[:, 3] / model.scale).square().mean()

    initial = loss().item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        error = loss()
        error.backward()
        optimizer.step()
    assert loss().item() < initial / 100


def test_register_vm_operations_and_bounds():
    model = oracle_primitives(20)
    programs = [Program("direct", 0), Program("direct", 1),
                Program("loop", 0, "zero", "a", "counter", "lt", "b", "state")]
    pairs = torch.tensor([[20, 20], [0, 20], [17, 5]])
    result, valid, steps = execute_programs(model, programs, pairs)
    assert valid.all()
    assert result.tolist() == [[40, 20, 22], [0, -20, 12], [400, 0, 85]]
    assert steps[2].tolist() == [20, 20, 5]
    # A generic counted loop program; no division instruction is registered.
    counted = Program("loop", 1, "a", "b", "state", "ge", "b", "counter")
    output, valid, steps = execute_programs(model, [counted], torch.tensor([[20, 1], [0, 1]]), budget=20)
    assert output.tolist() == [[20, 0]] and valid.all()
    assert steps.tolist() == [[20, 0]]
    assert not execute_programs(model, [counted], torch.tensor([[20, 1]]), budget=4)[1].item()
    # Zero divisor leads to a nonhalting hypothesis, not a made-up answer.
    assert not execute_programs(model, [counted], torch.tensor([[1, 0]]))[1].item()
    trace = trace_program(model, counted, 17, 5)
    assert trace["states"] == [{"state": 17, "counter": 0}, {"state": 12, "counter": 1},
                               {"state": 7, "counter": 2}, {"state": 2, "counter": 3}]
    assert trace["answer"] == 3 and trace["valid"]
    with pytest.raises(ValueError):
        execute_programs(model, programs, torch.tensor([[-1, 2]]))


@pytest.fixture(scope="module")
def inducer():
    torch.set_num_threads(1)
    return ProgramInducer(oracle_primitives(5), budget=8)


def test_generic_grammar_and_context_induction(inducer):
    assert len(enumerate_programs()) == 3104
    assert set(p.kind for p in inducer.programs) == {"direct", "loop"}
    assert set(p.instruction for p in inducer.programs) == {0, 1}
    before = {k: v.clone() for k, v in inducer.primitives.state_dict().items()}
    pairs, labels, supports = task_episodes(5, "floor_div", context_size=25)
    index = ((pairs[:, 0] == 5) & (pairs[:, 1] == 2)).nonzero().item()
    assert not (pairs[supports[index]] == pairs[index]).all(1).any()
    result = inducer.predict(pairs[supports[index]], labels[supports[index]], pairs[index].tolist())
    assert result["prediction"] == 2
    assert result["program"]["kind"] == "loop"
    assert result["program"]["instruction"] == 1
    assert result["program"]["returns"] == "counter"
    assert all(torch.equal(value, inducer.primitives.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in inducer.primitives.parameters())


def test_search_abstains_and_validates(inducer):
    assert inducer.predict([], torch.tensor([], dtype=torch.long), [1, 2])["prediction"] is None
    result = inducer.predict(torch.tensor([[0, 0]]), torch.tensor([0]), [3, 2])
    assert result["prediction"] is None
    result = inducer.predict(torch.tensor([[0, 0]]), torch.tensor([999]), [3, 2])
    assert result["reason"] == "no_consistent_program"
    with pytest.raises(ValueError, match="query"):
        inducer.predict(torch.tensor([[1, 2]]), torch.tensor([3]), [1, 2])
    with pytest.raises(ValueError, match="distinct"):
        inducer.predict(torch.tensor([[1, 2], [1, 2]]), torch.tensor([3, 3]), [3, 2])
    with pytest.raises(ValueError, match="integer"):
        inducer.predict([[1.5, 2.0]], torch.tensor([3]), [3, 2])


def test_one_hot_context_is_a_label_not_an_operation_dispatch(inducer):
    pairs, labels, supports = task_episodes(5, "mul", context_size=30)
    index = ((pairs[:, 0] == 4) & (pairs[:, 1] == 3)).nonzero().item()
    doc = {"examples": [{"a": int(pairs[i, 0]), "b": int(pairs[i, 1]),
                          "op": [0, 0, 0, 1], "result": int(labels[i])} for i in supports[index]],
           "query": {"a": 4, "b": 3, "op": [0, 0, 0, 1]}}
    first = inducer.predict_document(doc)
    assert first["prediction"] == 12  # fourth slot is not hardwired to division
    for row in [*doc["examples"], doc["query"]]:
        row["op"] = [1, 0, 0, 0]
    assert inducer.predict_document(doc) == first
    doc["query"]["result"] = 12
    with pytest.raises(ValueError, match="answer"):
        inducer.predict_document(doc)


def test_training_and_checkpoint_cli(tmp_path, monkeypatch, capsys):
    from teenie.program_train import main

    path = tmp_path / "primitives.pt"
    monkeypatch.setattr("sys.argv", ["train", "--device", "cpu", "--operand-max", "3",
                                    "--updates", "2", "--save", str(path)])
    main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    model, meta = load_primitives(tmp_path / "primitives.best.pt")
    assert meta["training_tasks"] == ["add", "sub", "mul"]
    assert model.operand_max == 3
    assert meta["instructions"] == ["add", "sub"]
    assert not model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_program_cuda_execution_and_checkpoint(tmp_path):
    model = oracle_primitives(5, "cuda")
    path = tmp_path / "cuda.pt"
    save_primitives(path, model, {"val": {"accuracy": 1}}, )
    restored, _ = load_primitives(path, "cuda")
    programs = [Program("loop", 0, "zero", "a", "counter", "lt", "b", "state")]
    output, valid, steps = execute_programs(restored, programs, torch.tensor([[5, 4], [0, 2]]))
    assert output.tolist() == [[20, 0]] and valid.all()
    assert steps.tolist() == [[4, 2]]
