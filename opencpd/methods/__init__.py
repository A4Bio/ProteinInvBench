# Copyright (c) CAIRI AI Lab. All rights reserved

from .alphadesign import AlphaDesign
from .esmif import ESMIF
from .gca import GCA
from .graphtrans import GraphTrans
from .gvp import GVP
from .proteinmpnn import ProteinMPNN
from .pifold import PiFold
from .structgnn import StructGNN

method_maps = {
    'pifold': PiFold,
    'alphadesign': AlphaDesign,
    'graphtrans': GraphTrans,
    'structgnn': StructGNN,
    'gvp': GVP,
    'gca': GCA,
    'proteinmpnn': ProteinMPNN,
    'esmif': ESMIF,
}

__all__ = [
    'method_maps', 'AlphaDesign', 'ESMIF', 'GraphTrans', 'GVP', 'ProteinMPNN',
    'PiFold', 'StructGNN',
]