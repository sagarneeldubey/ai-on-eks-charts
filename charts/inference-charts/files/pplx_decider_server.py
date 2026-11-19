"""HTTP adapter for the native pplx_decider library in a pinned model snapshot.

The native library owns protocol-v4 prompts, noncausal attention, decision
readouts, calibration and Locate; this module supplies the HTTP boundary.
"""

import argparse
import binascii
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException


def create_app(
    checkpoint: Path,
    model_name: str,
    *,
    device: str = "cuda",
    mode: str = "separate",
    max_length: int = 32768,
    batch_tokens: int = 32768,
    cpu_threads: int = 8,
    max_pixels: int | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The model snapshot includes the inference library that implements its
        # noncausal attention, readout head, calibrated probabilities and Locate.
        sys.path.insert(0, str(checkpoint / "source" / "src"))
        from pplx_decider import Decider, LengthError, RequestError
        from PIL import Image, UnidentifiedImageError

        app.state.request_errors = (
            RequestError,
            LengthError,
            binascii.Error,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        )
        app.state.decider = Decider.from_pretrained(
            checkpoint,
            device=device,
            mode=mode,
            max_length=max_length,
            batch_tokens=batch_tokens,
            cpu_threads=cpu_threads,
            max_pixels=max_pixels,
            fast=False,
        )
        yield
        del app.state.decider

    app = FastAPI(title="Perplexity Decider", lifespan=lifespan)

    @app.get("/health")
    async def health():
        # ASGI startup completes only after the checkpoint has loaded.
        return {"status": "ok", "model": model_name}

    @app.post("/v1/systemone")
    def decide(payload: dict[str, Any] | list[dict[str, Any]] = Body(...)):
        requests = [payload] if isinstance(payload, dict) else payload
        if not requests:
            raise HTTPException(status_code=422, detail="A batch must contain at least one request")
        for request in requests:
            images = request.get("images", [])
            # Native inference also accepts filesystem paths. HTTP clients may
            # send inline images only, so they cannot read files from the pod.
            if not isinstance(images, list) or any(
                not isinstance(image, str)
                or not image.startswith("data:image/")
                or ";base64," not in image
                for image in images
            ):
                raise HTTPException(status_code=422, detail="Images must be base64 data:image URLs")
        try:
            response = app.state.decider.decide(payload)
        except app.state.request_errors as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logging.exception("Decision inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from error
        if isinstance(response, list):
            return [{"model": model_name, **item} for item in response]
        return {"model": model_name, **response}

    return app


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("separate", "isolated"), default="separate")
    parser.add_argument("--max-length", type=positive_int, default=32768)
    parser.add_argument("--batch-tokens", type=positive_int, default=32768)
    parser.add_argument("--cpu-threads", type=positive_int, default=8)
    parser.add_argument("--max-pixels", type=positive_int)
    args = parser.parse_args()
    app = create_app(
        args.checkpoint,
        args.model_name,
        device=args.device,
        mode=args.mode,
        max_length=args.max_length,
        batch_tokens=args.batch_tokens,
        cpu_threads=args.cpu_threads,
        max_pixels=args.max_pixels,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
