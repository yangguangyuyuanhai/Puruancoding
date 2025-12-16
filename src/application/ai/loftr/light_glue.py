

class LightGlueONNX:

    def __init__(self, model_path="./models/dist_lightglue_pipeline.trt.onnx") -> None:
        pass

    def inferene(self, images, threshold=1000):
        return True, registry_image


class LightGlueTensorRT:

    def __init__(self) -> None:
        pass

    def inferene(self, images, threshold=1000):
        return True, registry_image


async def load_loftr_engine(engine_type="onnx", model_path=""):
    if engine_type == "onnx":
        return LightGlueONNX()
    elif engine_type == "tensorrt":
        return LightGlueTensorRT()
