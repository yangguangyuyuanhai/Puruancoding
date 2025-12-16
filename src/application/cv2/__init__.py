from . image_comparasion import Comparation
from . image_draw_border import BorderDrawer
from . image_dilation import Dilation
async def load_cv2_engines() -> tuple[Comparation, Dilation, BorderDrawer]:
    cv_comparasion = Comparation()
    cv_dilation = Dilation()
    cv_draw_rect = BorderDrawer()
    return cv_comparasion, cv_dilation, cv_draw_rect