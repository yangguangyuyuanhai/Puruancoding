import pathlib

from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
)


def convert_onnx_to_engine(model_path="../models/disk_lightglue_pipeline.trt.onnx"):
    _model_path = pathlib.Path(model_path)
    # for the gtx 1050ti, it should be false
    fp16 = False

    if _model_path.suffix == ".onnx":
        config = CreateConfig(fp16=fp16)
        onnx_network_frame = NetworkFromOnnxPath(str(_model_path))
        engine_from_network = EngineFromNetwork(onnx_network_frame, config)

        engine = SaveEngine(
            engine_from_network, str(_model_path.with_suffix(".engine"))
        )
        print("engine saved to {_model_path.with_suffix('.engine')}")
        engine()
        print("engine saved done")


def main():
    convert_onnx_to_engine()


if __name__ == "__main__":
    main()
