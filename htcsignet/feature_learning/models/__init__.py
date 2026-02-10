from .vit import vit_base_patch16_224_in21k

from .htcsignet import SigTransformer as htcsignet
from .signet_snn import SigNetSNN, SigNetSNN_Wavelet, SigNetSNN_thin_Wavelet

# 延迟导入可能有依赖问题的模块
try:
    from vanilla_vig.vig import vig_s_224_gelu
    from vanilla_vig.vig_snn import spiking_vig_ti_224
    _vig_available = True
except ImportError:
    _vig_available = False
    vig_s_224_gelu = None
    spiking_vig_ti_224 = None


available_models = {
                    'vit': vit_base_patch16_224_in21k,
                    'htcsignet': htcsignet,
                    'signet_snn': SigNetSNN,
                    'signet_snn_wavelet': SigNetSNN_Wavelet,
                    'signet_snn_thin_wavelet': SigNetSNN_thin_Wavelet,
                    }

if _vig_available:
    available_models['vig'] = vig_s_224_gelu
    available_models['vig_snn'] = spiking_vig_ti_224
