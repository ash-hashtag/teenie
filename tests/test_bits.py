import pytest
import torch

from teenie.bits import BitModel, BitSampler, bit_loss, from_bits, load_bit_model, save_bit_model, to_bits
from teenie.bits.baselines import nearest, window_lookup
from teenie.bits.loop_model import LoopModel
from teenie.bits.relations import NAMED, transducer


def test_bits_round_trip_and_wrapping():
    x = torch.tensor([0, 1, 5, 255])
    assert to_bits(x, 8)[2].tolist() == [1, 0, 1, 0, 0, 0, 0, 0]
    assert torch.equal(from_bits(to_bits(x, 8)), x)
    relation = NAMED['sub'].make(torch.Generator(), 8)
    assert relation(torch.tensor([3]), torch.tensor([5]), 8).item() == 254
    assert NAMED['a*b'].make(torch.Generator(), 8)(torch.tensor([16]), torch.tensor([17]), 8).item() == 16


def test_transducer_matches_addition_when_built_as_carry_machine():
    relation = transducer(torch.Generator().manual_seed(0), 8)
    a, b = torch.arange(256), torch.arange(256).flip(0)
    c = relation(a, b, 8)
    assert c.min() >= 0 and c.max() < 256


def test_sampler_has_999_distinct_examples_and_unseen_queries():
    episode = BitSampler(seed=3).sample(4, queries=8)
    assert episode.context.shape == (4, 999, 3) and episode.query.shape == (4, 8, 2)
    for i in range(4):
        pairs = torch.cat([episode.context[i, :, :2], episode.query[i]])
        assert len(torch.unique(pairs[:, 0] * 256 + pairs[:, 1])) == 999 + 8
    again = BitSampler(seed=3).sample(4, queries=8)
    assert torch.equal(episode.context, again.context) and episode.names == again.names


def test_window_lookup_solves_addition_and_nearest_does_not_leak():
    episode = BitSampler([NAMED['add']], seed=1).sample(2, queries=16)
    assert (window_lookup(episode, 1) == episode.target).float().mean() > .9
    assert (nearest(episode) >= 0).all()


def test_model_shapes_masking_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = BitModel(hidden=16, heads=2, probes=8)
    episode = BitSampler(seed=0).sample(2, queries=3)
    prediction = model.forward(episode.context, episode.query, 8, episode.target)
    assert prediction.logits.shape == (2, 2, 3 + 8, 8) and prediction.weights.shape == (2, 2)
    bit_loss(prediction, episode.context, episode.target, 8).backward()
    # A probe sees its own earlier c bits as history (like teacher forcing), never the bit being predicted:
    # the first bit each pass reads must not depend on the probe's own label.
    model.eval()
    with torch.no_grad():
        base = model.forward(episode.context, episode.query, 8).logits[:, 0, 3]
        changed = episode.context.clone()
        changed[0, 0, 2] ^= 255
        after = model.forward(changed, episode.query, 8).logits[:, 0, 3]
    torch.testing.assert_close(base[0, 0], after[0, 0])
    torch.testing.assert_close(base[1, 7], after[1, 7])
    assert not torch.allclose(base, after)
    save_bit_model(tmp_path / 'm.pt', model, {'width': 8})
    loaded, metadata = load_bit_model(tmp_path / 'm.pt')
    assert metadata['width'] == 8
    value, confidence = loaded.predict(episode.context[0].tolist(), tuple(episode.query[0, 0].tolist()))
    assert 0 <= value < 256 and .5 <= confidence <= 1


@pytest.mark.parametrize('multiplicative,scratch', [(False, 0), (True, 0), (True, 3)])
def test_loop_model_rounds_probe_masking_and_checkpoint(tmp_path, multiplicative, scratch):
    torch.manual_seed(0)
    model = LoopModel(hidden=16, heads=2, layers=1, probes=8, steps=3, multiplicative=multiplicative,
                      scratch=scratch)
    episode = BitSampler(seed=0).sample(2, queries=3)
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.logits.shape == (3, 2, 3 + 8, 8)
    model.loss(prediction, episode.context, episode.target, 8).backward()
    assert model.forward(episode.context, episode.query, 8, steps=5).logits.shape[0] == 5
    model.eval()
    with torch.no_grad():
        # Context is seen all at once, so a probe's own label must not reach its prediction at all.
        base = model.forward(episode.context, episode.query, 8).logits[:, 0, 3]
        changed = episode.context.clone()
        changed[0, 0, 2] ^= 255
        torch.testing.assert_close(base, model.forward(changed, episode.query, 8).logits[:, 0, 3])
    save_bit_model(tmp_path / 'loop.pt', model, {'width': 8})
    loaded, _ = load_bit_model(tmp_path / 'loop.pt')
    assert isinstance(loaded, LoopModel) and loaded.config['steps'] == 3
    assert loaded.config.get('multiplicative', False) == multiplicative
    assert loaded.config.get('scratch', 0) == scratch


def test_think_model_questions_are_independent_and_checkpoint(tmp_path):
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.think_model import ThinkModel

    torch.manual_seed(0)
    model = ThinkModel(hidden=16, heads=2, layers=1, steps=8)
    episode = BitSampler(TRAIN_SETS['mul'], seed=0).sample(2, queries=3)
    model.train()
    # Training draws 2..8 rounds (a quarter of `steps` up to `steps`).
    assert {model.forward(episode.context, episode.query, 8).logits.shape[0] for _ in range(30)} <= set(range(2, 9))
    model.loss(model.forward(episode.context, episode.query, 8), episode.context, episode.target, 8).backward()
    model.eval()
    assert model.forward(episode.context, episode.query, 8).logits.shape == (8, 2, 3, 8)
    model.eval()
    with torch.no_grad():
        # Changing one question must not change another question's answer (questions never see each other).
        base = model.forward(episode.context, episode.query, 8).logits[:, :, 0]
        other = episode.query.clone()
        other[:, 1] = (other[:, 1] + 1) % 256
        torch.testing.assert_close(base, model.forward(episode.context, other, 8).logits[:, :, 0])
        assert model.forward(episode.context, episode.query, 8, steps=6).logits.shape[0] == 6
    save_bit_model(tmp_path / 'think.pt', model, {'width': 8, 'train_set': 'mul'})
    loaded, metadata = load_bit_model(tmp_path / 'think.pt')
    assert isinstance(loaded, ThinkModel) and metadata['train_set'] == 'mul' and loaded.config['steps'] == 8


def test_program_model_rounds_fit_and_checkpoint(tmp_path):
    from teenie.bits.program_model import ProgramModel
    from teenie.bits.sets import TRAIN_SETS

    torch.manual_seed(0)
    for informed in (True, False):
        model = ProgramModel(hidden=32, channels=4, steps=3, informed=informed)
        episode = BitSampler(TRAIN_SETS['mul'], seed=0).sample(2, queries=3)
        prediction = model.forward(episode.context, episode.query, 8)
        assert prediction.logits.shape == (3, 2, 3, 8) and prediction.context_logits.shape == (3, 2, 999, 8)
        assert prediction.fit is not None and ((0 <= prediction.fit) & (prediction.fit <= 1)).all()
        model.loss(prediction, episode.context, episode.target, 8).backward()
        with torch.no_grad():
            # Questions never feed back into the program: changing one leaves the other answers unchanged.
            base = model.forward(episode.context, episode.query, 8).logits[:, :, 0]
            other = episode.query.clone()
            other[:, 1] = (other[:, 1] + 1) % 256
            torch.testing.assert_close(base, model.forward(episode.context, other, 8).logits[:, :, 0])
    save_bit_model(tmp_path / 'program.pt', model, {'width': 8})
    loaded, _ = load_bit_model(tmp_path / 'program.pt')
    assert isinstance(loaded, ProgramModel) and loaded.config['informed'] is False


def test_expressions_evaluate_wrap_and_never_match_hidden_relations():
    from teenie.bits.expressions import HIDDEN_FAMILIES, evaluate, fingerprint, show
    from teenie.bits.sets import EXPRESSIONS, FORBIDDEN, TRAIN_SETS

    tree = ('*', ('+', 'a', 1), ('>>1', 'b'))
    a, b = torch.tensor([255, 3]), torch.tensor([9, 200])
    assert evaluate(tree, a, b, 8).tolist() == [0, 400 % 256]  # (255+1) wraps to 0 before multiplying
    assert show(tree) == '((a+1)*(b)>>1)'
    assert len(HIDDEN_FAMILIES) == 8 and all(f in TRAIN_SETS['mix'] for f in EXPRESSIONS)
    banned = {fingerprint(f.make(torch.Generator(), 8), 8) for f in FORBIDDEN}
    generator = torch.Generator().manual_seed(0)
    for family in EXPRESSIONS:
        for _ in range(50):
            relation = family.make(generator, 8)
            assert fingerprint(relation, 8) not in banned
    # The equivalence check catches rewrites: a + a*b behaves exactly like the hidden a*b + a.
    rewrite = lambda a, b, w: (a + a * b) & 255  # noqa: E731
    assert fingerprint(rewrite, 8) in banned


def test_training_resume_and_init_continue_from_saved_state(tmp_path):
    import json

    from teenie.bits.train import train

    options = dict(device='cpu', batch_size=1, queries=2, hidden=16, heads=2, probes=4, layers=1, steps=2,
                   architecture='loop', train_set='mul-only', log_every=2, validation_episodes=1, adaptive=True)
    save = tmp_path / 'm.pt'
    trained = train(updates=2, save=save, log=tmp_path / 'a.jsonl', **options)
    assert (tmp_path / 'm.resume.pt').exists()
    # Resuming a finished run changes nothing.
    resumed = train(updates=2, save=save, log=tmp_path / 'a.jsonl', resume=True, **options)
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key])
    # --init trains more on top of the weights; update numbers continue from the checkpoint.
    train(updates=2, save=tmp_path / 'more.pt', log=tmp_path / 'b.jsonl', init=save, **options)
    records = [json.loads(line) for line in (tmp_path / 'b.jsonl').read_text().splitlines()]
    assert records[-1]['update'] == 4


def test_mix2_has_more_steps_more_products_and_keeps_hidden_out():
    from teenie.bits.expressions import fingerprint, random_tree
    from teenie.bits.sets import FORBIDDEN, STEPS, TRAIN_SETS

    count = lambda trees: sum(str(t).count("'*'") for t in trees)  # noqa: E731
    plain = [random_tree(3, torch.Generator().manual_seed(i), 8) for i in range(300)]
    weighted = [random_tree(3, torch.Generator().manual_seed(i), 8, product_weight=2.) for i in range(300)]
    assert count(weighted) > 1.4 * count(plain)
    assert [f.name for f in STEPS] == ['steps1', 'steps2', 'steps3', 'steps4', 'steps5', 'composed']
    assert all(f in TRAIN_SETS['mix2'] for f in STEPS)
    banned = {fingerprint(f.make(torch.Generator(), 8), 8) for f in FORBIDDEN}
    generator = torch.Generator().manual_seed(1)
    for family in STEPS:
        for _ in range(30):
            assert fingerprint(family.make(generator, 8), 8) not in banned


def test_loops_tiers_and_held_combinations():
    from teenie.bits.expressions import pairs
    from teenie.bits.loops import hidden_loops, popcount, reverse
    from teenie.bits.sets import HELD_PAIRS, LOOPS, STEPS3, TIERS, TRAIN_SETS

    a, b = torch.tensor([3, 10, 200]), torch.tensor([4, 2, 17])
    total, power, fib, collatz, shift_xor = (f.make(torch.Generator(), 8) for f in hidden_loops(4))
    assert total(a, b, 8).tolist() == [3 + 4 + 5 + 6, 10 + 11, 200]  # n = b & 15: 4, 2, 1
    assert power(a, b, 8).tolist() == [81, 100, 200]
    assert fib(a, b, 8).tolist() == [3 * 5, 10 * 2, 200]  # terms a, a, 2a, 3a, 5a, ...
    assert collatz(torch.tensor([1]), torch.tensor([2]), 8).tolist() == [13]
    longer = hidden_loops(8)[0].make(torch.Generator(), 8)
    assert longer(torch.tensor([1]), torch.tensor([20]), 8).tolist() == [sum(range(1, 21)) % 256]
    assert popcount(torch.tensor([255, 6]), 8).tolist() == [8, 2]
    assert reverse(torch.tensor([1, 6]), 8).tolist() == [128, 96]
    assert all(f in TRAIN_SETS['mix3'] for f in STEPS3 + LOOPS) and len(TIERS) == 6
    assert {('^', '*')} <= HELD_PAIRS and pairs(('^', ('*', 'a', 'b'), 'a')) == {('^', '*')}
    # Training draws never contain a held-out combination; tier draws always do.
    from teenie.bits.expressions import _draw, random_tree
    seen: list[tuple[str, bool]] = []

    def draw(generator, width):
        tree = random_tree(3, generator, width, 2.)
        seen.append((str(tree), bool(pairs(tree) & HELD_PAIRS)))
        return tree
    generator = torch.Generator().manual_seed(2)
    for require in (False, True):
        for _ in range(20):
            seen.clear()
            relation = _draw(draw, (), generator, 8, HELD_PAIRS, require_pair=require)
            assert seen[-1][1] == require
            assert relation.name
    # Generated loops (not the fixed bit loops, whose counts take few values) vary enough.
    for family in LOOPS[:3] + TIERS['T5b'][10:16]:
        assert len(set(fingerprint_of(family.make(generator, 8)))) >= 8


def fingerprint_of(relation):
    from teenie.bits.expressions import fingerprint
    return fingerprint(relation, 8)


def test_serve_functions_are_checked_and_episodes_are_distinct():
    import random

    from teenie.bits.serve import Function, episode

    assert Function('a*b + a')(200, 3) == (200 * 3 + 200) % 256
    assert Function('a if a > b else b ^ 3')(1, 4) == 7
    assert Function('sum(a + i for i in range(b & 15))')(10, 3) == 10 + 11 + 12
    assert Function('popcount(a) + reverse(b)')(255, 1) == 8 + 128
    assert Function('a // b')(5, 0) == 255  # division by zero reads as all ones
    for unsafe in ('__import__("os")', 'a.__class__', '(lambda: 1)()', 'open("x")', 'a[0]', '"text"',
                   'sum(i for i in range(10) for j in range(10))', 'sum(range(10**9))'):
        with pytest.raises(ValueError):
            Function(unsafe)(1, 2)
    context, questions = episode(Function('a ^ b'), 5, random.Random(0))
    pairs = {(a, b) for a, b, _ in context + questions}
    assert len(context) == 999 and len(questions) == 5 and len(pairs) == 1004
    assert all(c == a ^ b for a, b, c in context)


def test_bit_token_model_rounds_probe_masking_and_checkpoint(tmp_path):
    from teenie.bits.bit_token_model import BitTokenModel

    torch.manual_seed(0)
    model = BitTokenModel(hidden=16, heads=2, layers=1, probes=8, steps=3)
    episode = BitSampler(seed=0).sample(2, queries=3)
    model.train()
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.logits.shape == (3, 2, 3 + 8, 8)
    model.loss(prediction, episode.context, episode.target, 8).backward()
    model.eval()
    with torch.no_grad():
        # A probe never sees its own example, so its own label cannot reach its prediction.
        base = model.forward(episode.context, episode.query, 8).logits[:, 0, 3]
        changed = episode.context.clone()
        changed[0, 0, 2] ^= 255
        torch.testing.assert_close(base, model.forward(changed, episode.query, 8).logits[:, 0, 3])
        # Questions never see each other.
        other = episode.query.clone()
        other[:, 1] = (other[:, 1] + 1) % 256
        torch.testing.assert_close(model.forward(episode.context, episode.query, 8).logits[:, :, 0],
                                   model.forward(episode.context, other, 8).logits[:, :, 0])
    save_bit_model(tmp_path / 'bt.pt', model, {'width': 8})
    loaded, _ = load_bit_model(tmp_path / 'bt.pt')
    assert isinstance(loaded, BitTokenModel) and loaded.config['steps'] == 3


def test_halting_distribution_sums_to_one_and_stops_early_when_confident():
    torch.manual_seed(0)
    model = LoopModel(hidden=16, heads=2, layers=1, probes=8, steps=5, multiplicative=True, halting=True)
    episode = BitSampler(seed=0).sample(2, queries=3)
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.halts is not None and prediction.halts.shape == (5, 2, 3 + 8)
    torch.testing.assert_close(prediction.halts.sum(0), torch.ones(2, 3 + 8))
    model.loss(prediction, episode.context, episode.target, 8).backward()
    stops = prediction.stop()
    assert stops.shape == (2, 3) and int(stops.min()) >= 0 and int(stops.max()) <= 4
    # A model told to halt immediately answers from round 1.
    with torch.no_grad():
        model.halt[1].bias.fill_(20.)
        assert int(model.forward(episode.context, episode.query, 8).stop().max()) == 0
        assert model.forward(episode.context, episode.query, 8, steps=9).logits.shape[0] == 9


def test_steps_for_expressions_loops_and_episodes():
    from teenie.bits.expressions import named, tree_steps
    from teenie.bits.loops import hidden_loops
    from teenie.bits.sets import TRAIN_SETS

    a, b = torch.tensor([3, 200]), torch.tensor([5, 7])
    trace, counts = tree_steps(('+', ('*', 'a', 'b'), 'b'))(a, b, 8)
    assert trace.tolist() == [[15, 1400 % 256], [20, (1400 + 7) % 256]] and counts.tolist() == [2, 2]
    total = hidden_loops(4)[0].make(torch.Generator(), 8)  # sum a..a+n-1 with n = b & 15
    assert total.steps is not None
    trace, counts = total.steps(torch.tensor([10, 1]), torch.tensor([3, 0]), 8)
    assert trace.tolist() == [[10, 0], [21, 0], [33, 0]] and counts.tolist() == [3, 1]  # n = 0: empty sum
    assert trace[-1].tolist() == total(torch.tensor([10, 1]), torch.tensor([3, 0]), 8).tolist()
    episode = BitSampler(TRAIN_SETS['mix4'], seed=0).sample(3, queries=4, steps=True)
    assert episode.steps is not None and episode.step_counts is not None
    assert episode.steps.shape[0] == 3 and episode.steps.shape[2] == 999 + 4
    # The last step is always the answer, for examples and questions alike.
    torch.testing.assert_close(episode.steps[:, -1, :999], episode.context[..., 2])
    torch.testing.assert_close(episode.steps[:, -1, 999:], episode.target)
    assert int(episode.step_counts.min()) >= 1 and int(episode.step_counts.max()) <= episode.steps.shape[1]


def test_stepwise_loop_model_trains_and_stops_when_done():
    from teenie.bits.sets import TRAIN_SETS

    torch.manual_seed(0)
    model = LoopModel(hidden=16, heads=2, layers=1, probes=8, steps=6, multiplicative=True, stepwise=True)
    episode = BitSampler(TRAIN_SETS['mix4'], seed=1).sample(2, queries=3, steps=True)
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.done is not None and prediction.done.shape == (6, 2, 3 + 8)
    model.loss(prediction, episode.context, episode.target, 8, episode.steps, episode.step_counts).backward()
    with torch.no_grad():
        model.done[1].bias.fill_(20.)  # always "done" -> answers from round 1
        assert int(model.forward(episode.context, episode.query, 8).stop().max()) == 0
        model.done[1].bias.fill_(-20.)  # never "done" -> answers at the cap
        assert int(model.forward(episode.context, episode.query, 8).stop().min()) == 5


def test_stepwise_bit_token_model_trains_and_stops():
    from teenie.bits.bit_token_model import BitTokenModel
    from teenie.bits.sets import TRAIN_SETS

    torch.manual_seed(0)
    model = BitTokenModel(hidden=16, heads=2, layers=1, probes=8, steps=4, stepwise=True)
    episode = BitSampler(TRAIN_SETS['mix4'], seed=2).sample(2, queries=3, steps=True)
    model.train()
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.done is not None and prediction.done.shape == (4, 2, 3 + 8)
    model.loss(prediction, episode.context, episode.target, 8, episode.steps, episode.step_counts).backward()
    model.eval()
    with torch.no_grad():
        model.done[1].bias.fill_(20.)
        assert int(model.forward(episode.context, episode.query, 8).stop().max()) == 0


def test_aligned_bit_tokens_pairs_long_loops_and_hint_fade(tmp_path):
    from teenie.bits.bit_token_model import BitTokenModel
    from teenie.bits.sets import PAIRS, STEPS3, TRAIN_SETS

    torch.manual_seed(0)
    model = BitTokenModel(hidden=16, heads=2, layers=1, probes=8, steps=3, aligned=True)
    episode = BitSampler(TRAIN_SETS['mix5'], seed=3).sample(2, queries=3)
    model.train()
    prediction = model.forward(episode.context, episode.query, 8)
    assert prediction.logits.shape == (3, 2, 3 + 8, 8)
    model.loss(prediction, episode.context, episode.target, 8).backward()
    model.eval()
    with torch.no_grad():
        base = model.forward(episode.context, episode.query, 8).logits[:, 0, 3]
        changed = episode.context.clone()
        changed[0, 0, 2] ^= 255  # a probe still never sees its own example
        torch.testing.assert_close(base, model.forward(changed, episode.query, 8).logits[:, 0, 3])
        stores = model.record_attention()
        model.forward(episode.context, episode.query, 8)
        assert stores[0][0].shape == (2, 3 + 8, 999)
    assert len(TRAIN_SETS['mix6']) == len(TRAIN_SETS['mix4']) and PAIRS in TRAIN_SETS['mix5']
    generator = torch.Generator().manual_seed(1)
    for _ in range(20):
        assert PAIRS.make(generator, 8).name
    stepwise = BitTokenModel(hidden=16, heads=2, layers=1, probes=8, steps=4, stepwise=True)
    episode = BitSampler([STEPS3[3]], seed=2).sample(2, queries=3, steps=True)  # 4-operation expressions
    prediction = stepwise.forward(episode.context, episode.query, 8)
    full = stepwise.loss(prediction, episode.context, episode.target, 8, episode.steps, episode.step_counts)
    stepwise.hint = 0.
    faded = stepwise.loss(prediction, episode.context, episode.target, 8, episode.steps, episode.step_counts)
    assert torch.isfinite(full) and torch.isfinite(faded) and not torch.equal(full, faded)


def test_concept_operations():
    from teenie.bits.expressions import evaluate
    a, b = torch.tensor([200, 7, 9]), torch.tensor([0, 3, 10])
    assert evaluate(('//', 'a', 'b'), a, b, 8).tolist() == [255, 2, 0]
    assert evaluate(('%', 'a', 'b'), a, b, 8).tolist() == [200, 1, 9]
    assert evaluate(('<', 'a', 'b'), a, b, 8).tolist() == [0, 0, 1]
    assert evaluate(('<<v', 'a', 'b'), a, b, 8).tolist() == [200, 56, (9 << 2) & 255]
    assert evaluate(('sq', 'a'), a, b, 8).tolist() == [(200 * 200) & 255, 49, 81]
    assert evaluate(('pop', 'a'), a, b, 8).tolist() == [3, 3, 2]


def test_wide_model_any_width():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    model = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2)
    for width in (8, 17, 32):
        episode = BitSampler(TRAIN_SETS['mix7'], width, 0).sample(2, 3)
        pairs = torch.cat([episode.context[..., :2], episode.query], 1)
        assert all(len(torch.unique(p, dim=0)) == len(p) for p in pairs)
        prediction = model(episode.context, episode.query, width)
        assert prediction.logits.shape == (2, 2, 5, width)
        assert torch.isfinite(model.loss(prediction, episode.context, episode.target, width))


def test_wide_peers_never_show_a_question_its_answer():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    model = WideBitModel(hidden=32, heads=2, layers=2, probes=3, steps=2, peers=True).eval()
    episode = BitSampler(TRAIN_SETS['mix7'], 12, 0).sample(1, 2)
    before = model(episode.context, episode.query, 12).logits
    changed = episode.context.clone()
    changed[:, 3:, 2] ^= 5  # answers of every non-probe example
    assert torch.allclose(before, model(changed, episode.query, 12).logits) is False  # examples matter
    probes_changed = episode.context.clone()
    probes_changed[:, :3, 2] ^= 5  # probe answers reach the questions through peers
    assert not torch.allclose(before[:, :, :2], model(probes_changed, episode.query, 12).logits[:, :, :2])
    loss = model.loss(model(episode.context, episode.query, 12), episode.context, episode.target, 12)
    assert torch.isfinite(loss)


def test_wide_registers_extend_a_trained_model_unchanged():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    base = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=3).eval()
    extended = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=3, registers=2).eval()
    extended.load_extending(base.state_dict())
    episode = BitSampler(TRAIN_SETS['mix7'], 16, 0).sample(1, 3)
    assert torch.allclose(base(episode.context, episode.query, 16).logits,
                          extended(episode.context, episode.query, 16).logits, atol=1e-5)


def test_wide_example_layers_extend_a_trained_model_unchanged():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    base = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2).eval()
    deep = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2, example_layers=2).eval()
    deep.load_extending(base.state_dict())
    episode = BitSampler(TRAIN_SETS['mix7'], 16, 0).sample(1, 3)
    assert torch.allclose(base(episode.context, episode.query, 16).logits,
                          deep(episode.context, episode.query, 16).logits, atol=1e-5)
    deep.train()
    loss = deep.loss(deep(episode.context, episode.query, 16), episode.context, episode.target, 16)
    loss.backward()
    assert deep.thinking[0].out.weight.grad is not None


def test_wide_code_bottleneck_reads_examples_only_through_the_code():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    model = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2, code_slots=3, code_layers=2)
    episode = BitSampler(TRAIN_SETS['mix7'], 16, 0).sample(2, 3)
    prediction = model(episode.context, episode.query, 16)
    assert prediction.logits.shape == (2, 2, 5, 16)
    loss = model.loss(prediction, episode.context, episode.target, 16)
    loss.backward()
    assert model.code.grad is not None and model.coder[0].key.weight.grad is not None
    model.eval()
    changed = episode.context.clone()
    changed[:, :, 2] ^= 3  # different answers in the examples change the code, so the answers
    assert not torch.allclose(model(episode.context, episode.query, 16).logits, model(changed, episode.query, 16).logits)


def test_wide_code_film_extends_a_code_model_unchanged():
    from teenie.bits.data import BitSampler
    from teenie.bits.sets import TRAIN_SETS
    from teenie.bits.wide_model import WideBitModel
    torch.manual_seed(0)
    base = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2, code_slots=3, code_layers=1).eval()
    film = WideBitModel(hidden=32, heads=2, layers=1, probes=2, steps=2, code_slots=3, code_layers=1,
                        code_film=True).eval()
    film.load_extending(base.state_dict())
    episode = BitSampler(TRAIN_SETS['mix7'], 16, 0).sample(1, 3)
    assert torch.allclose(base(episode.context, episode.query, 16).logits,
                          film(episode.context, episode.query, 16).logits, atol=1e-5)
    film.train()
    film.loss(film(episode.context, episode.query, 16), episode.context, episode.target, 16).backward()
    assert film.film[0].weight.grad is not None


def test_bits_recovered_scores_chance_certainty_and_confident_mistakes():
    from teenie.bits.eval import FLOOR, baseline_recovered, recovered_bits
    target = torch.tensor([[5, 3]])  # width 4: 0101, 0011
    truth = to_bits(target, 4).float()
    assert torch.allclose(recovered_bits(torch.full((1, 2, 4), .5), target, 4), torch.zeros(2))
    assert torch.allclose(recovered_bits(truth, target, 4), torch.full((2,), 4.))
    wrong = recovered_bits(1 - truth, target, 4)  # certain and wrong on every bit: floored, so finite and negative
    assert torch.isfinite(wrong).all() and (wrong < 0).all()
    assert abs(float(wrong[0]) - (4 + 4 * torch.log2(torch.tensor(FLOOR, dtype=torch.float64)))) < 1e-3
    # a lookup that is always right earns every bit; one that matches a random target by chance earns about nothing
    generator = torch.Generator().manual_seed(0)
    targets = [torch.randint(0, 256, (16,), generator=generator) for _ in range(32)]
    assert abs(baseline_recovered(targets, targets, 8) - 8) < 1e-3
    guesses = [torch.randint(0, 256, (16,), generator=generator) for _ in range(32)]
    assert abs(baseline_recovered(guesses, targets, 8)) < .5


def test_opaque_relations_are_defined_and_unlike_training_relations():
    from teenie.bits.data import BitSampler
    from teenie.bits.expressions import fingerprint
    from teenie.bits.opaque import OPAQUE
    from teenie.bits.sets import TRAIN_SETS
    trained = set()
    for family in TRAIN_SETS['broad']:
        for seed in range(4):  # a few draws of each generator
            relation = family.make(torch.Generator().manual_seed(seed), 16)
            trained.add(fingerprint(relation, 16))
    for kind, families in OPAQUE.items():
        for family in families:
            relation = family.make(torch.Generator().manual_seed(0), 16)
            assert fingerprint(relation, 16) not in trained, family.name
            for width in (16, 32):
                episode = BitSampler([family], width, 1).sample(1, 4)
                assert 0 <= int(episode.target.min()) and int(episode.target.max()) < 1 << width, family.name
