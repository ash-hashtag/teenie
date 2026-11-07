import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from teenie import ContextModel, EpisodeSampler, FunctionTask, load_model, save_model
from teenie.data import DEFAULT_TASKS, div, function_value, load_tasks, mul
from teenie.eval import COMPOSITE, evaluate_function
from teenie.metrics import measure, training_loss
from teenie.model import Prediction
from teenie.numeric import F32_MAX, F32_MIN, decode, encode, kinds
from teenie.train import train


def test_default_four_functions_and_ieee_semantics():
    assert [t.name for t in DEFAULT_TASKS] == ['add', 'sub', 'mul', 'div']
    division = FunctionTask('div', div)
    assert function_value(division, 1, 0) == float('inf')
    assert math.isnan(function_value(division, 0, 0))
    assert function_value(division, F32_MIN, F32_MAX) == 0
    assert function_value(FunctionTask('mul', mul), F32_MAX, 2) == float('inf')
    assert math.isnan(function_value(COMPOSITE, 0, 0))
    assert function_value(COMPOSITE, 0, 1) == float('inf')
    assert function_value(COMPOSITE, 2, 4) == 6
    assert function_value(COMPOSITE, F32_MAX, 1) == float('inf')
    # Float32 intermediate overflow differs from evaluating everything in float64.
    task = FunctionTask('intermediate', lambda a, b: (a*a) / b)
    assert function_value(task, 1e20, 1e20) == float('inf')


def test_full_range_sampler_is_reproducible_unique_and_held_out():
    episode = EpisodeSampler(seed=5).sample(8)
    same = EpisodeSampler(seed=5).sample(8)
    torch.testing.assert_close(episode.context, same.context, equal_nan=True)
    assert not (episode.context[..., :2] == episode.query[:, None]).all(-1).any()
    pairs = episode.context[..., :2]
    assert torch.isfinite(pairs).all() and (pairs >= 0).all()
    assert (pairs == F32_MAX).any() and (pairs == 0).any()
    assert ((pairs > 0) & (pairs < 1e-30)).any() and (pairs > 1e30).any()
    for row in pairs:
        assert len(row.unique(dim=0)) == 999


def test_numeric_encoding_roundtrip_and_special_kinds():
    values = torch.tensor([0., F32_MIN, 1e-30, -1e-20, 1., -1., 1e20, F32_MAX,
                           -F32_MAX, float('inf'), -float('inf'), float('nan')])
    assert encode(values).isfinite().all()
    restored = decode(encode(values), kinds(values))
    torch.testing.assert_close(restored, values, rtol=3e-5, atol=0, equal_nan=True)
    assert kinds(values).tolist()[-3:] == [1, 2, 3]
    assert decode(torch.tensor([1e10]), torch.tensor([0])).isfinite().all()


def test_invalid_inputs_and_real_python_errors():
    with pytest.raises(ValueError):
        FunctionTask('bad', div, a_range=(-1, 2))
    def broken(a, b):
        raise RuntimeError('intentional')
    with pytest.raises(RuntimeError, match='intentional'):
        EpisodeSampler([FunctionTask('bad', broken)]).sample()
    model = ContextModel(knots=16)
    episode = EpisodeSampler().sample(2)
    with pytest.raises(ValueError, match='float32'):
        model.forward(episode.context.double(), episode.query.double())
    with pytest.raises(ValueError, match='finite'):
        model.forward(episode.context, episode.query * float('inf'))
    with pytest.raises(ValueError, match='steps'):
        model.forward(episode.context, episode.query, steps=0)


def test_nonfinite_support_is_safe_and_loop_has_gradients():
    torch.set_num_threads(1)
    model = ContextModel(knots=16, steps=3)
    episode = EpisodeSampler(seed=4).sample(4)
    episode.context[:, :3, 2] = torch.tensor([float('inf'), -float('inf'), float('nan')])
    prediction = model.forward(episode.context, episode.query, return_steps=True)
    assert prediction.encoded.shape == (4, 3)
    assert prediction.encoded.isfinite().all() and prediction.kind_logits.isfinite().all()
    target = torch.tensor([2., float('inf'), -float('inf'), float('nan')])
    loss = training_loss(prediction, target)
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert model.centers.grad.abs().sum() > 0
    reversed_output = model.forward(episode.context.flip(1), episode.query)
    torch.testing.assert_close(reversed_output.encoded, prediction.encoded[:, -1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(prediction.encoded[:, 0], prediction.encoded[:, -1])


def test_all_nonfinite_loss_and_decode():
    target = torch.tensor([float('inf'), -float('inf'), float('nan')])
    logits = torch.eye(4)[kinds(target)] * 10
    prediction = Prediction(torch.zeros(3, requires_grad=True), logits.requires_grad_())
    loss = training_loss(prediction, target)
    assert loss.isfinite()
    loss.backward()
    assert prediction.encoded.grad.isfinite().all()
    torch.testing.assert_close(prediction.values(), target, equal_nan=True)


def test_metrics_separate_nonfinite_and_finite_errors():
    class FakeModel(ContextModel):
        def forward(self, context, query, **kwargs):
            # Correct infinity and undefined, incorrect finite value.
            return Prediction(encode(torch.tensor([100., 0., 0.])), torch.eye(4)[torch.tensor([0, 1, 3])] * 10)
    from teenie.data import Episode
    episode = Episode(torch.zeros(3, 2, 3), torch.zeros(3, 2), torch.tensor([1., float('inf'), float('nan')]))
    report = measure(FakeModel(knots=8), [episode])
    assert report['finite_accuracy'] == 0 and report['nonfinite_accuracy'] == 1
    assert report['undefined_count'] == 1 and report['pos_inf_count'] == 1
    assert report['accuracy'] == pytest.approx(2/3)
    json.dumps(report, allow_nan=False)


def test_checkpoint_and_no_inference_weight_updates(tmp_path):
    model = ContextModel(knots=16)
    episode = EpisodeSampler(seed=8).sample(1)
    path = tmp_path / 'model.pt'
    save_model(path, model, {'training_functions': ['add', 'sub', 'mul', 'div']})
    restored, metadata = load_model(path)
    torch.testing.assert_close(restored.forward(episode.context, episode.query).encoded,
                               model.forward(episode.context, episode.query).encoded)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    evaluate_function(restored, COMPOSITE, count=8)
    assert all(torch.equal(v, before[k]) for k, v in restored.state_dict().items())
    with pytest.raises(ValueError, match='query'):
        restored.predict(episode.context[0].tolist(), episode.context[0, 0, :2].tolist())
    torch.save({'format': 'teenie-float-loop-v1'}, path)
    with pytest.raises(ValueError, match='retrain'):
        load_model(path)


def test_typed_callable_train_and_eval_cli(tmp_path):
    functions = tmp_path / 'functions.py'
    functions.write_text("from teenie import FunctionTask\nTASKS=[FunctionTask('custom',lambda a,b: a+b)]\n")
    path = tmp_path / 'v2.pt'
    train(load_tasks(functions), device='cpu', updates=2, knots=8, batch_size=2,
          save=path, validation_batches=1)
    assert path.exists() and path.with_name('v2.best.pt').exists()
    report = tmp_path / 'report.json'
    subprocess.run([sys.executable, '-m', 'teenie.eval', '--load', str(path), '--device', 'cpu',
                    '--count', '4', '--functions', str(functions),
                    '--output', str(report)], check=True, capture_output=True)
    assert json.loads(report.read_text())['training_functions'] == ['custom']


def test_fixed_episode_learning():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = ContextModel(knots=16, steps=2)
    from teenie.families import bounded_basics
    episode = EpisodeSampler(bounded_basics(), seed=5).sample_queries(4, queries=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    def loss():
        return training_loss(model.forward(episode.context, episode.query), episode.target)
    initial = loss().item()
    for _ in range(120):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    assert loss().item() < initial * .9


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_gradient_and_cpu_checkpoint(tmp_path):
    model = ContextModel(knots=16).cuda()
    episode = EpisodeSampler(seed=9).sample(3).to('cuda')
    prediction = model.forward(episode.context, episode.query)
    training_loss(prediction, episode.target).backward()
    assert model.centers.grad.isfinite().all()
    path = tmp_path / 'cuda.pt'
    save_model(path, model)
    restored, _ = load_model(path)
    torch.testing.assert_close(restored.forward(episode.context.cpu(), episode.query.cpu()).encoded,
                               prediction.encoded.cpu(), rtol=1e-4, atol=1e-5)


def test_fixed_context_999_enforced_everywhere(tmp_path):
    from teenie import CONTEXT_SIZE
    assert CONTEXT_SIZE == 999
    episode = EpisodeSampler(seed=12).sample(1)
    assert episode.context.shape == (1, 999, 3)
    model = ContextModel(knots=8)
    for count in (1, 99, 998, 1000):
        context = torch.zeros(1, count, 3)
        with pytest.raises(ValueError, match='999'):
            model.forward(context, episode.query)
    with pytest.raises(TypeError):
        EpisodeSampler().sample(context_size=99)
    path = tmp_path / 'fixed.pt'
    save_model(path, model)
    payload = torch.load(path, weights_only=True)
    assert payload['context_size'] == 999
    payload['context_size'] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match='999'):
        load_model(path)


def test_fixed_boundary_probes_cover_undefined_without_query_leakage():
    from teenie.eval import boundary_episodes
    episode = boundary_episodes(COMPOSITE)[0]
    assert episode.context.shape == (16, 999, 3)
    assert not (episode.context[..., :2] == episode.query[:, None]).all(-1).any()
    assert torch.isnan(episode.target).any() and torch.isposinf(episode.target).any()
    assert torch.isfinite(episode.target).any()


def test_curriculum_and_generated_families_have_no_heldout_structure():
    import random
    from teenie.families import TRAIN_FAMILIES, curriculum_max, family_factory, heldout_tasks, make_family
    assert [curriculum_max(n, 900) for n in (1, 300, 301, 600, 601, 900)] == [16.,16.,64.,64.,256.,256.]
    rng = random.Random(123)
    names = {family_factory()(rng).name for _ in range(200)}
    assert set(TRAIN_FAMILIES) <= names
    assert {'square_plus_ratio', 'cross_ratio'}.isdisjoint(names)
    assert {'recurrence_depth8', 'recurrence_depth16'} <= {t.name for t in heldout_tasks()}
    for family in TRAIN_FAMILIES:
        task = make_family(random.Random(2), family)
        assert math.isfinite(function_value(task, 2, 3))
    episode = EpisodeSampler(seed=22, task_factory=family_factory(16)).sample(2)
    assert episode.context.shape == (2,999,3)
    assert episode.context[..., :2].max() <= 16
    assert not (episode.context[..., :2] == episode.query[:,None]).all(-1).any()


def test_memory_states_direct_values_and_all_loop_budgets():
    from teenie.families import bounded_basics
    model = ContextModel(knots=16)
    episode = EpisodeSampler(bounded_basics(), seed=42).sample(2)
    for budget in (1, 4, 8, 16):
        output = model.forward(episode.context, episode.query, steps=budget)
        assert output.finite_values is not None and output.output_scale is not None
        assert output.finite_values.isfinite().all()
    loss = training_loss(output, episode.target)
    loss.backward()
    assert model.centers.grad.abs().sum() > 0
    trace = model.forward(episode.context, episode.query, steps=4, return_steps=True)
    assert trace.finite_values.shape == (2,4)
    assert trace.output_scale.shape == (2,4)
    torch.testing.assert_close(trace.finite_values[:,-1], model.forward(episode.context, episode.query, steps=4).finite_values)


def test_nearest_baseline_uses_context_only():
    from teenie.data import Episode
    context = torch.zeros(1,999,3)
    context[0,:,0] = torch.arange(999)
    context[0,:,2] = torch.arange(999) * 2
    episode = Episode(context, torch.tensor([[5.1,0.]]), torch.tensor([10.]))
    assert measure(ContextModel(knots=8), [episode], baseline='nearest')['accuracy'] == 1
    episode.target[:] = 9999
    assert measure(ContextModel(knots=8), [episode], baseline='nearest')['accuracy'] == 0


def test_multiquery_support_separation_and_shared_fit():
    from teenie.families import bounded_basics
    episode = EpisodeSampler(bounded_basics(), seed=44).sample_queries(2, queries=16)
    assert episode.context.shape == (2, 999, 3)
    assert episode.query.shape == (2, 16, 2)
    assert episode.target.shape == (2, 16)
    assert measure(ContextModel(steps=8), [episode])['count'] == 32
    assert measure(ContextModel(steps=8), [episode], baseline='nearest')['count'] == 32
    assert measure(ContextModel(steps=8), [episode], baseline='mean')['count'] == 32
    assert not (episode.context[:, :, None, :2] == episode.query[:, None]).all(-1).any()
    for row in episode.query:
        assert row.unique(dim=0).shape[0] == 16
    model = ContextModel(steps=32)
    result = model.forward(episode.context, episode.query)
    separate = model.forward(episode.context, episode.query[:, 3])
    torch.testing.assert_close(result.values()[:, 3], separate.values())
    trace = model.forward(episode.context, episode.query, return_steps=True)
    assert trace.finite_values.shape == (2, 16, 32)
    assert trace.kind_logits.shape == (2, 16, 32, 4)
    torch.testing.assert_close(trace.finite_values[..., -1], result.finite_values)
    assert training_loss(trace, episode.target).isfinite()
    assert (trace.support_residual[:, -1] < trace.support_residual[:, 0]).all()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        result = model.forward(episode.context, episode.query)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_all_special_support_and_constant_support_are_stable():
    from teenie.families import bounded_basics
    episode = EpisodeSampler(bounded_basics(), seed=81).sample_queries(2, queries=16)
    model = ContextModel()
    for value in (0., 7., float('inf'), float('nan')):
        episode.context[..., 2] = value
        prediction = model.forward(episode.context, episode.query)
        assert prediction.finite_values.isfinite().all()
        assert training_loss(prediction, episode.target).isfinite()
        if math.isfinite(value):
            torch.testing.assert_close(prediction.values(), torch.full_like(episode.target, value))
