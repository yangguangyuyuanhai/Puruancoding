from application.ai.loftr.light_glue import load_loftr_engine
from application.ai.unet.image_mask import load_unet_mask_engine
from application.cv2 import load_cv2_engines

from helps import vairables as app_vars
from helps.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class AppState:

    def __init__(self):
        self._loftr_engine = self.build_loftr_engine()
        self._unet_engine = self.build_unet_engine()
        self.cv_comprasion, self.cv_dilation, self.cv_border_drawer = self.build_cv_engines()
    def build_loftr_engine(self):
        if self._loftr_engine is None:
            self._loftr_engine = load_loftr_engine(
                app_vars.APP_LoFTR_MODEL_ENGINE, app_vars.APP_LoFTR_MODEL_PATH
            )
        return self._loftr_engine
    
    def build_unet_engine(self):
        if self._unet_engine is None:
            self._unet_engine = load_unet_mask_engine(
                app_vars.APP_UNET_MODEL_ENGINE, app_vars.APP_UNET_MODEL_PATH
            )
        return self._unet_engine

    async def build_cv_engines(self):
         return load_cv2_engines()
    async def get_loftr_engine(self):
         return self._loftr_engine
    async def get_unet_engine(self):
        return self._unet_engine

    async def get_cv_comparasion(self):
        return self.cv_comprasion

    async def get_cv_dilation(self):
        return self.cv_dilation
    async def get_cv_draw_rect(self):
        return self.cv_draw_rect
