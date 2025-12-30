from .vit import vit_base_patch16_224_in21k

from .htcsignet import SigTransformer as htcsignet
from vanilla_vig.vig import vig_s_224_gelu
from vanilla_vig.vig_snn import spiking_vig_ti_224


available_models = {
                    'vit': vit_base_patch16_224_in21k,
                    'htcsignet': htcsignet,
                    'vig': vig_s_224_gelu,
                    'vig_snn': spiking_vig_ti_224,
                    }
