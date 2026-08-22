#!/usr/bin/env python3
"""End-to-end live verification of FireRedAudio WebUI server and inference queue."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    port = args.port
    base_url = f"http://127.0.0.1:{port}"
    print(f"Testing WebUI endpoints at {base_url}...", flush=True)

    # 1. Test Root SPA HTML
    with urllib.request.urlopen(f"{base_url}/") as response:
        html = response.read().decode("utf-8")
        assert response.status == 200, "Root page did not return 200"
        assert "FireRedAudio" in html, "Root page does not contain FireRedAudio"
        print("✓ Root SPA HTML page served successfully", flush=True)

    # 2. Test System Status
    with urllib.request.urlopen(f"{base_url}/api/v1/system/status") as response:
        status_data = json.loads(response.read().decode("utf-8"))
        assert "status" in status_data, "Missing status field"
        assert "memory" in status_data, "Missing memory field"
        print(f"✓ System status checked (Model status: {status_data['status']}, Metal Active: {status_data['memory']['active_memory_gb']} GB)", flush=True)

    # 3. Test Presets API
    with urllib.request.urlopen(f"{base_url}/api/v1/system/presets") as response:
        presets = json.loads(response.read().decode("utf-8"))
        assert "balanced" in presets and "fast" in presets, "Presets missing balanced/fast"
        print("✓ Quality presets API verified", flush=True)

    # 4. Test Project & Asset Upload
    req = urllib.request.Request(
        f"{base_url}/api/v1/projects",
        data=json.dumps({"name": "E2E Test Project", "description": "Auto verification"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        proj = json.loads(response.read().decode("utf-8"))
        proj_id = proj["id"]
        print(f"✓ Created test project #{proj_id}", flush=True)

    # 5. Submit real ASR job
    wav_path = "assets/examples/asr_zh_fleurs.wav"
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    # Use requests-like multipart or standard boundary upload
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"test_asr.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + (
        f"\r\n--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"project_id\"\r\n\r\n"
        f"{proj_id}\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    upload_req = urllib.request.Request(
        f"{base_url}/api/v1/assets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(upload_req) as response:
        asset = json.loads(response.read().decode("utf-8"))
        asset_id = asset["id"]
        print(f"✓ Uploaded audio asset #{asset_id} (duration: {asset['duration']}s)", flush=True)

    # Submit ASR job
    job_req = urllib.request.Request(
        f"{base_url}/api/v1/jobs",
        data=json.dumps({
            "task": "asr",
            "project_id": proj_id,
            "params": {"audio_asset_ids": [asset_id]},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(job_req) as response:
        job = json.loads(response.read().decode("utf-8"))
        job_id = job["id"]
        print(f"✓ Submitted ASR Job #{job_id}, polling for completion...", flush=True)

    # Poll job until finished
    t0 = time.time()
    job_done = False
    while time.time() - t0 < 120:
        time.sleep(1.0)
        with urllib.request.urlopen(f"{base_url}/api/v1/jobs/{job_id}") as response:
            j = json.loads(response.read().decode("utf-8"))
            if j["status"] == "completed":
                transcript = j["result"]["transcript"]
                print(f"✓ Job #{job_id} COMPLETED in {j['latency_seconds']}s! Transcript: '{transcript}'", flush=True)
                assert "浪漫主义" in transcript, "Transcript content does not match expected result"
                job_done = True
                break
            elif j["status"] == "failed":
                raise AssertionError(f"Job failed: {j['error']}")
            else:
                print(f"  ...status: {j['status']} - {j['current_step']}", flush=True)

    if not job_done:
        raise TimeoutError(f"Job #{job_id} did not finish within timeout period.")

    print("\n🎉 ALL WEBUI E2E INTEGRATION CHECKS PASSED!", flush=True)

if __name__ == "__main__":
    main()
