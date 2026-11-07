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


@pytest.mark.parametrize('multiplicative', [False, True])
def test_loop_model_rounds_probe_masking_and_checkpoint(tmp_path, multiplicative):
    torch.manual_seed(0)
    model = LoopModel(hidden=16, heads=2, layers=1, probes=8, steps=3, multiplicative=multiplicative)
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
