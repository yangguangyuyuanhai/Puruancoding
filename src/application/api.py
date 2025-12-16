import uvicorn
from fastapi import Depends, FastAPI

from helps import vairables as app_vars
from helps.logger import AppLogger

from .app_state import AppState
from .entities.api import ImageRequestEntry

app = FastAPI()
logger = AppLogger.get_logger(__name__)
_app_state = AppState()


def get_app_state() -> AppState:
    return _app_state

def encode_image_to_base64(image):
    return base64.b64encode(image).decode("utf-8")

@app.post("/registry")
async def registry(
    image_entry: ImageRequestEntry,
    state=Depends(get_app_state),
):
    area = image_entry.area
    enableColor = image_entry.enableColor
    registry_threshold = image_entry.registry_threshold
    images = image_entry.images
    logger.info(f"area: {area}, enableColor: {enableColor}, images: {images}")
    # 对图片进行配准
    loftr_engine = state.get_loftr_engine()
    is_registry, registry_image = await loftr_engine.inferene(images, registry_threshold)

    if is_registry == False:
        return {"message": "图片无法进行配准"}
    
    # 对图片进行UNET分割
    image_1, image_2 = images
    unet_engine = state.get_unet_engine()
    image1_mask = await unet_engine.inferene(image_1)
    registry_mask = await unet_engine.inferene(registry_image)

    #对图片进行膨胀或者腐蚀
    cv_dilation = await state.get_cv_dilation()
    image1_mask = cv_dilation.do_dilation(image1_mask)
    registry_mask = cv_dilation.do_dilation(registry_mask)

    #对图片进行对比
    cv_comparison = await state.get_cv_comparasion()
    image_comp_marix = await cv_comparison.do_comparasion(image1_mask, registry_mask)

    #用对比后的坐标 
    cv_draw_border = await state.get_cv_draw_rect()
    final_image = cv_draw_border.do_draw(image_comp_marix)
    ecoded_image = encode_image_to_base64(final_image)
    return {"image": ecoded_image}


@app.get("/healthcheck")
async def health_check(state=Depends(get_app_state)):
    return {"status": "ok"}


async def start():
    config = uvicorn.Config(
        app=app,
        host=app_vars.APP_WEBAPI_HOST,
        port=app_vars.APP_WEBAPI_PORT,
        log_level=logger.level,
        reload=True,
    )
    server = uvicorn.Server(config)
    await server.serve()
