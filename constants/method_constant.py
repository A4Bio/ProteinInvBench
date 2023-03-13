from methods import StructGNN_Design, GraphTrans_Design, GVP_Design, GCA_Design, AlphaDesign_Design, ESMIF_Design, ProteinMPNN_Design, PiFold

method_maps = {
    'PiFold': PiFold,
    'AlphaDesign': AlphaDesign_Design,
    'GraphTrans': GraphTrans_Design,
    'StructGNN': StructGNN_Design,
    'GVP': GVP_Design,
    'GCA': GCA_Design,
    'ProteinMPNN': ProteinMPNN_Design,
    'ESMIF': ESMIF_Design,
}