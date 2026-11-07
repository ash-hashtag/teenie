"""Bit-vector relations: 999 (a, b, c) examples, predict every bit of c for a new (a, b)."""
from .checkpoint import load_bit_model, save_bit_model
from .data import BitEpisodes, BitSampler, from_bits, to_bits
from .model import BitModel, BitPrediction, bit_loss
from .relations import Family, Relation, HELDOUT_FAMILIES, STRESS_FAMILIES, TRAIN_FAMILIES

__all__ = ['BitEpisodes', 'BitModel', 'BitPrediction', 'BitSampler', 'Family', 'HELDOUT_FAMILIES', 'Relation',
           'STRESS_FAMILIES', 'TRAIN_FAMILIES', 'bit_loss', 'from_bits', 'load_bit_model', 'save_bit_model',
           'to_bits']
