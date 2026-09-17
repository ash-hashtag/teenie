import json
import subprocess
import sys

import pytest

from teenie.relation_generate import generate_document
from teenie.relation_model import RelationModel, predict_document


def test_generate_unique_held_out_and_repeatable():
    options = dict(valid=lambda a, b: b != 0, context_size=419, query=(17, 5))
    document, answer = generate_document(lambda a, b: a // b, **options)
    assert (document, answer) == generate_document(lambda a, b: a // b, **options)
    pairs = {(r['a'], r['b']) for r in document['examples']}
    assert len(pairs) == 419 and (17, 5) not in pairs
    assert all(r['c'] == r['a'] // r['b'] for r in document['examples'])
    assert document['query'] == {'a': 17, 'b': 5}
    assert answer['expected'] == 3


def test_custom_real_relation_and_validation():
    document, answer = generate_document(lambda a, b: a + b / 2, query=(1, 3))
    assert answer['expected'] == 2.5
    assert predict_document(RelationModel(), document)['context_size'] == 99
    with pytest.raises(ValueError, match='valid pairs'):
        generate_document(lambda a, b: a, context_size=441)
    with pytest.raises(ValueError, match='valid pair'):
        generate_document(lambda a, b: a, query=(21, 0))
    with pytest.raises(ValueError, match='finite real'):
        generate_document(lambda a, b: float('nan'))
    with pytest.raises(ZeroDivisionError):
        generate_document(lambda a, b: a / b, context_size=440)


def test_generator_cli(tmp_path):
    function = tmp_path / 'relation.py'
    function.write_text('def f(a,b): return (a-7)**2 + 2*b\n')
    output = tmp_path / 'input.json'
    subprocess.run([sys.executable, '-m', 'teenie.relation_generate', '--function', str(function),
                    '--query', '17', '5', '--output', str(output)], check=True)
    document = json.loads(output.read_text())
    answer = json.loads(output.with_suffix('.answer.json').read_text())
    assert answer['expected'] == 110
    assert 'expected' not in document and 'c' not in document['query']
    assert len(document['examples']) == 99
