"""Test the HTTP contract without downloading model weights or requiring a GPU."""

import importlib.util
import binascii
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import UnidentifiedImageError
import anyio

SERVER = Path(__file__).resolve().parents[1] / "files" / "pplx_decider_server.py"
SPEC = importlib.util.spec_from_file_location("pplx_decider_server", SERVER)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class RequestError(ValueError):
    pass


class LengthError(ValueError):
    pass


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.decider = Mock()
        self.loader = Mock(return_value=self.decider)
        native = types.ModuleType("pplx_decider")
        native.Decider = types.SimpleNamespace(from_pretrained=self.loader)
        native.RequestError = RequestError
        native.LengthError = LengthError
        self.modules = patch.dict(sys.modules, {"pplx_decider": native})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        original_path = sys.path.copy()
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.app = server.create_app(
            Path("/models/checkpoint"),
            "perplexity-ai/pplx-decider-v1.3",
            max_length=8192,
            batch_tokens=8192,
            cpu_threads=4,
        )
        self.request = {
            "state": "The traffic light is red.",
            "questions": {
                "color": {"type": "choice", "criteria": {"red": None, "blue": None}}
            },
        }
        self.response = {
            "answers": {"color": {"choice": "red", "probabilities": {"red": 0.9, "blue": 0.1}}},
            "usage": {"input_tokens": 20, "output_tokens": 0, "sequences": 1},
        }
        self.decider.decide.return_value = self.response

    def test_loads_native_reference_path_before_health_is_ready(self):
        with TestClient(self.app) as client:
            self.loader.assert_called_once_with(
                Path("/models/checkpoint"),
                device="cuda",
                mode="separate",
                max_length=8192,
                batch_tokens=8192,
                cpu_threads=4,
                max_pixels=None,
                fast=False,
            )
            result = client.get("/health")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "ok")

    def test_preserves_native_decisions_probabilities_and_usage(self):
        with TestClient(self.app) as client:
            result = client.post("/v1/systemone", json=self.request)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(
                result.json(), {"model": "perplexity-ai/pplx-decider-v1.3", **self.response}
            )
            self.decider.decide.assert_called_once_with(self.request)

    def test_preserves_batch_order_and_response_shape(self):
        second = {**self.response, "answers": {"color": {"choice": "blue"}}}
        self.decider.decide.return_value = [self.response, second]
        with TestClient(self.app) as client:
            payload = [self.request, {**self.request, "state": "The traffic light is blue."}]
            result = client.post("/v1/systemone", json=payload)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(
                [item["answers"]["color"]["choice"] for item in result.json()], ["red", "blue"]
            )
            self.decider.decide.assert_called_once_with(payload)

    def test_passes_native_question_types_and_inline_images_without_rewriting(self):
        payload = {
            "state": "Alice is blocked by a 401 error.",
            "questions": {
                "blocked": {"type": "noul"},
                "urgency": {"type": "score", "criteria": ["low", "high"]},
            },
            "images": ["data:image/png;base64,aW1hZ2U="],
        }
        with TestClient(self.app) as client:
            self.assertEqual(client.post("/v1/systemone", json=payload).status_code, 200)
            self.decider.decide.assert_called_once_with(payload)
            self.decider.decide.reset_mock()
            locate = {
                "state": payload["state"],
                "questions": {"person": {"type": "locate"}},
            }
            self.assertEqual(client.post("/v1/systemone", json=locate).status_code, 200)
            self.decider.decide.assert_called_once_with(locate)

    def test_native_schema_length_and_image_errors_return_422(self):
        for error in (
            RequestError("Invalid questions"),
            LengthError("Input exceeds length limit"),
            binascii.Error("Invalid image base64"),
            UnidentifiedImageError("Invalid image format"),
        ):
            with self.subTest(error=error), TestClient(self.app) as client:
                self.decider.decide.side_effect = error
                result = client.post("/v1/systemone", json=self.request)
                self.assertEqual(result.status_code, 422)
                self.assertEqual(result.json()["detail"], str(error))

    def test_rejects_image_paths_and_urls_before_native_inference(self):
        for image in ("/models/private.png", "https://example.com/image.png"):
            with self.subTest(image=image), TestClient(self.app) as client:
                payload = {**self.request, "images": [image]}
                self.assertEqual(client.post("/v1/systemone", json=payload).status_code, 422)
        self.decider.decide.assert_not_called()

    def test_rejects_empty_batches_and_invalid_json_shapes(self):
        with TestClient(self.app) as client:
            for payload in ([], "invalid", [1], {"images": "invalid"}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post("/v1/systemone", json=payload).status_code, 422)
        self.decider.decide.assert_not_called()

    def test_inference_failure_returns_500_without_exposing_internal_error(self):
        self.decider.decide.side_effect = RuntimeError("internal model information")
        with TestClient(self.app) as client, self.assertLogs(level="ERROR"):
            result = client.post("/v1/systemone", json=self.request)
            self.assertEqual(result.status_code, 500)
            self.assertEqual(result.json(), {"detail": "Inference failed"})

    def test_failed_model_load_fails_server_startup(self):
        self.loader.side_effect = RuntimeError("Model could not load")
        with self.assertRaisesRegex(RuntimeError, "Model could not load"):
            with TestClient(self.app):
                self.fail("The server must not become ready after a failed model load")

    def test_health_stays_responsive_when_inference_occupies_all_thread_workers(self):
        started, release = Event(), Event()

        def block_inference(_):
            started.set()
            if not release.wait(10):
                raise RuntimeError("Test inference was not released")
            return self.response

        self.decider.decide.side_effect = block_inference
        with TestClient(self.app) as client:
            async def limit_workers():
                limiter = anyio.to_thread.current_default_thread_limiter()
                original = limiter.total_tokens
                limiter.total_tokens = 1
                return limiter, original

            limiter, original = client.portal.call(limit_workers)
            with ThreadPoolExecutor(max_workers=2) as executor:
                inference = executor.submit(client.post, "/v1/systemone", json=self.request)
                try:
                    self.assertTrue(started.wait(5))
                    health = executor.submit(client.get, "/health").result(timeout=2)
                    self.assertEqual(health.status_code, 200)
                finally:
                    release.set()
                    inference.result(timeout=5)
                    client.portal.call(setattr, limiter, "total_tokens", original)


if __name__ == "__main__":
    unittest.main()
