"""Integration tests for FireRedAudio WebUI backend and API endpoints."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from fireredaudio_mlx.webui.app import create_app
from fireredaudio_mlx.webui.workspace import WorkspaceStore

class WebUIAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="firered_webui_test_")
        cls.store = WorkspaceStore(root_dir=Path(cls.tmp_dir))
        cls.app = create_app(workspace_dir=cls.tmp_dir)
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_01_root_serves_spa_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("FireRedAudio Studio", res.text)

    def test_02_system_status_and_presets(self):
        res = self.client.get("/api/v1/system/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("memory", data)

        pres_res = self.client.get("/api/v1/system/presets")
        self.assertEqual(pres_res.status_code, 200)
        presets = pres_res.json()
        self.assertIn("balanced", presets)
        self.assertIn("fast", presets)
        self.assertIn("high_quality", presets)

    def test_03_project_crud(self):
        # Create
        create_res = self.client.post("/api/v1/projects", json={"name": "测试项目", "description": "用于单元测试"})
        self.assertEqual(create_res.status_code, 200)
        proj = create_res.json()
        self.assertEqual(proj["name"], "测试项目")
        proj_id = proj["id"]

        # Detail
        get_res = self.client.get(f"/api/v1/projects/{proj_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["id"], proj_id)

        # List
        list_res = self.client.get("/api/v1/projects")
        self.assertEqual(list_res.status_code, 200)
        self.assertTrue(any(p["id"] == proj_id for p in list_res.json()))

        # Delete
        del_res = self.client.delete(f"/api/v1/projects/{proj_id}")
        self.assertEqual(del_res.status_code, 200)

    def test_04_asset_upload_and_retrieval(self):
        wav_path = "assets/examples/asr_zh_fleurs.wav"
        with open(wav_path, "rb") as f:
            content = f.read()

        upload_res = self.client.post(
            "/api/v1/assets/upload",
            files={"file": ("test_upload.wav", content, "audio/wav")},
            data={"source": "test"},
        )
        self.assertEqual(upload_res.status_code, 200)
        asset = upload_res.json()
        self.assertEqual(asset["name"], "test_upload.wav")
        self.assertGreater(asset["duration"], 0)
        self.assertTrue(asset["media_url"].startswith("/api/v1/media/assets/"))

        # Media streaming endpoint
        media_res = self.client.get(asset["media_url"])
        self.assertEqual(media_res.status_code, 200)
        self.assertEqual(media_res.headers["content-type"], "audio/wav")

    def test_05_voice_profile_management(self):
        wav_path = "assets/examples/tts_zh_prompt.wav"
        with open(wav_path, "rb") as f:
            content = f.read()
        asset = self.client.post(
            "/api/v1/assets/upload",
            files={"file": ("prompt.wav", content, "audio/wav")},
        ).json()

        voice_res = self.client.post("/api/v1/voices", json={
            "name": "播音员小红",
            "prompt_text": "同时，他强调微调要科学有序。",
            "audio_asset_id": asset["id"],
            "language": "zh",
            "description": "标准女声",
        })
        self.assertEqual(voice_res.status_code, 200)
        v = voice_res.json()
        self.assertEqual(v["name"], "播音员小红")

        list_v = self.client.get("/api/v1/voices").json()
        self.assertTrue(any(x["id"] == v["id"] for x in list_v))

    def test_06_job_submission_and_queue_query(self):
        job_res = self.client.post("/api/v1/jobs", json={
          "task": "asr",
          "params": {"prompt": "Transcribe speech to text."},
        })
        self.assertEqual(job_res.status_code, 200)
        job = job_res.json()
        self.assertEqual(job["task"], "asr")
        self.assertIn(job["status"], ["queued", "loading", "preprocessing", "inferencing"])

        # Query jobs list
        jobs_list = self.client.get("/api/v1/jobs").json()
        self.assertTrue(any(j["id"] == job["id"] for j in jobs_list))

    def test_07_project_patch_update(self):
        proj = self.client.post("/api/v1/projects", json={"name": "原名称", "description": "原描述"}).json()
        patch_res = self.client.patch(f"/api/v1/projects/{proj['id']}", json={"name": "新名称", "description": "更新后的描述"})
        self.assertEqual(patch_res.status_code, 200)
        updated = patch_res.json()
        self.assertEqual(updated["name"], "新名称")
        self.assertEqual(updated["description"], "更新后的描述")

    def test_08_media_streaming_security_and_by_id(self):
        wav_path = "assets/examples/asr_zh_fleurs.wav"
        with open(wav_path, "rb") as f:
            content = f.read()
        asset = self.client.post(
            "/api/v1/assets/upload",
            files={"file": ("secure_test.wav", content, "audio/wav")},
        ).json()

        # 1. Access by asset ID
        res_id = self.client.get(f"/api/v1/media/{asset['id']}")
        self.assertEqual(res_id.status_code, 200)
        self.assertEqual(res_id.headers["content-type"], "audio/wav")

        # 2. Reject non-audio files (e.g. metadata JSON)
        res_json = self.client.get(f"/api/v1/media/assets/{asset['id']}.json")
        self.assertEqual(res_json.status_code, 404)

    def test_09_invalid_upload_rejected(self):
        # Non-audio text content
        bad_res = self.client.post(
            "/api/v1/assets/upload",
            files={"file": ("corrupt.wav", b"THIS_IS_NOT_AUDIO", "audio/wav")},
        )
        self.assertEqual(bad_res.status_code, 400)
        self.assertIn("无法解析为有效音频", bad_res.json()["detail"])

    def test_10_voice_reference_protection_on_asset_delete(self):
        wav_path = "assets/examples/tts_zh_prompt.wav"
        with open(wav_path, "rb") as f:
            content = f.read()
        asset = self.client.post(
            "/api/v1/assets/upload",
            files={"file": ("ref_prompt.wav", content, "audio/wav")},
        ).json()

        voice = self.client.post("/api/v1/voices", json={
            "name": "保护测试声音",
            "prompt_text": "测试文本",
            "audio_asset_id": asset["id"],
            "language": "zh",
        }).json()

        # Attempt to delete referenced asset
        del_res = self.client.delete(f"/api/v1/assets/{asset['id']}")
        self.assertEqual(del_res.status_code, 400)
        self.assertIn("已被声音模板", del_res.json()["detail"])

        # Delete voice first then delete asset
        self.assertEqual(self.client.delete(f"/api/v1/voices/{voice['id']}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/v1/assets/{asset['id']}").status_code, 200)

    def test_11_job_retry_and_result_endpoint(self):
        job = self.client.post("/api/v1/jobs", json={
            "task": "asr",
            "params": {"prompt": "Transcribe speech to text."},
        }).json()

        # Result endpoint
        res_endpoint = self.client.get(f"/api/v1/results/{job['id']}")
        self.assertEqual(res_endpoint.status_code, 200)
        self.assertEqual(res_endpoint.json()["id"], job["id"])

        # Cancel job then retry it
        self.client.post(f"/api/v1/jobs/{job['id']}/cancel")
        retry_res = self.client.post(f"/api/v1/jobs/{job['id']}/retry")
        self.assertEqual(retry_res.status_code, 200)
        retried_job = retry_res.json()
        self.assertEqual(retried_job["task"], job["task"])
        self.assertEqual(retried_job["status"], "queued")

    def test_12_model_status_alias(self):
        res = self.client.get("/api/v1/model/status")
        self.assertEqual(res.status_code, 200)
        self.assertIn("memory", res.json())

if __name__ == "__main__":
    unittest.main()
