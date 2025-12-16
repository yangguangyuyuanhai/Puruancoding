class UNetMaskONNX:

    def __init__(self):
        pass

    async def inferene(self, image):
        pass


class UNetMaskTensorRT:

    def __init__(self):
        pass

    async def inferene(self, image):
        pass


async def load_unet_mask_engine(
    engine, model_path="./models/dist_lightglue_pipeline.trt.onnx"
):
    if engine == "onnx":
        return UNetMaskONNX()
    elif engine == "tensorrt":
        return UNetMaskTensorRT()
