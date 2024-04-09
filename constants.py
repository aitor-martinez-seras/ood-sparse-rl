from pathlib import Path

# OOD methods
# TODO: Voy a usar estas listas para definir el tipo de metodo que es cada uno
#     y asi poder reutilizar codigo entre ellos. Por ejemplo, todos los CONV_METHODS
#     van a tener que trabajar con feature maps, y todos los distance methods requieren
#     generar un "vector de normalidad o in distribution" o varios. De la misma manera,
#OOD_METHODS = ['msp', 'energy','pca', 'l1', 'l2']
LOGITS_METHODS      = ['msp', 'energy']
DISTANCE_METHODS    = ['l1', 'l2', 'l1_per_action', 'l2_per_action']
CONV_METHODS        = ['pca'] + DISTANCE_METHODS
IM_MOD_METHODS      = ['forward_dynamics_l1', 'forward_dynamics_l2']
OOD_METHODS         = CONV_METHODS + LOGITS_METHODS + IM_MOD_METHODS

# Paths
PATH_OOD_DIR = Path('./ood_storage')
PATH_PLOTS = Path('./plots')