"""Exercise the local HTTP contract without altering any model artifacts."""

from http.client import HTTPConnection
import json
import threading
import unittest
from unittest.mock import Mock, patch

from interface.server import LocalServer, metadata
from src.freeze_model import PROJECT_ROOT


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = Mock(return_value={"prediction": {"predicted_outcome": "home_win"}})
        cls.server = LocalServer(("127.0.0.1", 0), predict=cls.predict)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.predict.reset_mock()
        self.predict.side_effect = None

    def request(self, method="GET", path="/", body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, payload=None, **kwargs):
        return self.request("POST", "/api/predict", json.dumps(payload or {"home": "Arsenal", "away": "Chelsea", "date": "2026-09-12"}), {"Content-Type": "application/json", **kwargs})

    def test_static_assets_and_security_headers(self):
        for path in ("/", "/app.js", "/styles.css"):
            status, headers, body = self.request(path=path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_no_project_or_raw_file_serving(self):
        for path in ("/data/raw/manifest.json", "/../README.md", "/.env", "/models/model_b_xgb.json", "/%2e%2e/README.md"):
            self.assertEqual(self.request(path=path)[0], 404)

    def test_valid_prediction_calls_existing_action(self):
        status, _, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["prediction"]["predicted_outcome"], "home_win")
        self.predict.assert_called_once_with("Arsenal", "Chelsea", "2026-09-12", root=PROJECT_ROOT)

    def test_foreign_origin_and_host_rejected(self):
        for _ in range(10):
            for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}, {"Origin": "null"}):
                self.assertEqual(self.post(**headers)[0], 403)
        self.predict.assert_not_called()

    def test_same_origin_accepted(self):
        self.assertEqual(self.post(Origin=f"http://127.0.0.1:{self.server.server_port}")[0], 200)

    def test_bad_json_and_oversized_body_rejected(self):
        for body in ("[", "[]", "{}", '"string"', "x" * 2049, '{"home":1,"away":"Chelsea","date":"2026-09-12"}', '{"home":"","away":"Chelsea","date":"2026-09-12"}'):
            self.assertEqual(self.request("POST", "/api/predict", body, {"Content-Type": "application/json"})[0], 400)
        self.predict.assert_not_called()

    def test_non_json_rejected(self):
        self.assertEqual(self.request("POST", "/api/predict", "home=Arsenal", {"Content-Type": "text/plain"})[0], 415)
        self.predict.assert_not_called()

    def test_busy_request_rejected(self):
        with self.server.busy:
            self.assertEqual(self.post()[0], 429)
        self.predict.assert_not_called()

    def test_prediction_errors_are_actionable_and_unlock(self):
        self.predict.side_effect = ValueError("Choose two different teams.")
        status, _, body = self.post()
        self.assertEqual(status, 400)
        self.assertIn("different teams", json.loads(body)["error"])
        self.assertFalse(self.server.busy.locked())

    def test_missing_data_fails_without_local_path_disclosure(self):
        self.predict.side_effect = OSError("private local path")
        status, _, body = self.post()
        self.assertEqual(status, 503)
        self.assertNotIn("private local path", body.decode())

    def test_metadata_and_failure(self):
        with patch("interface.server.metadata", return_value={"teams": ["Arsenal"]}):
            status, _, body = self.request(path="/api/metadata")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["teams"], ["Arsenal"])
        with patch("interface.server.metadata", side_effect=RuntimeError("broken freeze")):
            self.assertEqual(self.request(path="/api/metadata")[0], 503)

    def test_real_metadata_canonical_teams_and_snapshot(self):
        value = metadata()
        self.assertEqual(value["snapshot_date"], "2026-05-24")
        self.assertEqual(value["minimum_date"], "2026-05-25")
        self.assertIn("Arsenal", value["teams"])
        self.assertNotIn("Barcelona", value["teams"])


if __name__ == "__main__":
    unittest.main()
