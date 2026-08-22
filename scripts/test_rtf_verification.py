import urllib.request
import json
import time

base = "http://127.0.0.1:7860/api/v1"

# 1. Upload ASR test audio
with open("assets/examples/asr_zh_fleurs.wav", "rb") as f:
    wav_bytes = f.read()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
crlf = "\r\n"
body = (
    f"--{boundary}{crlf}"
    f"Content-Disposition: form-data; name=\"file\"; filename=\"test_rtf_asr.wav\"{crlf}"
    f"Content-Type: audio/wav{crlf}{crlf}"
).encode("utf-8") + wav_bytes + (
    f"{crlf}--{boundary}--{crlf}"
).encode("utf-8")

req = urllib.request.Request(f"{base}/assets/upload", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
asset_asr = json.loads(urllib.request.urlopen(req).read().decode())
print(f"Uploaded ASR asset #{asset_asr['id']}, duration: {asset_asr['duration']}s")

# 2. Submit ASR job
job_req = urllib.request.Request(
    f"{base}/jobs",
    data=json.dumps({"task": "asr", "params": {"audio_asset_ids": [asset_asr["id"]]}, "preset": "balanced"}).encode(),
    headers={"Content-Type": "application/json"}
)
job_asr = json.loads(urllib.request.urlopen(job_req).read().decode())
print(f"Submitted ASR Job #{job_asr['id']}, waiting...")

while True:
    time.sleep(1.0)
    j = json.loads(urllib.request.urlopen(f"{base}/jobs/{job_asr['id']}").read().decode())
    if j["status"] == "completed":
        print(f"✓ ASR Completed! Latency: {j['latency_seconds']}s | RTF: {j['result'].get('rtf')} | Step: {j['current_step']}")
        assert "rtf" in j["result"], "RTF missing in ASR result"
        break
    elif j["status"] == "failed":
        raise AssertionError(f"ASR Job failed: {j['error']}")

# 3. Upload TTS prompt audio
with open("assets/examples/tts_zh_prompt.wav", "rb") as f:
    tts_bytes = f.read()

body_tts = (
    f"--{boundary}{crlf}"
    f"Content-Disposition: form-data; name=\"file\"; filename=\"test_rtf_prompt.wav\"{crlf}"
    f"Content-Type: audio/wav{crlf}{crlf}"
).encode("utf-8") + tts_bytes + (
    f"{crlf}--{boundary}--{crlf}"
).encode("utf-8")

req_tts = urllib.request.Request(f"{base}/assets/upload", data=body_tts, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
asset_tts = json.loads(urllib.request.urlopen(req_tts).read().decode())

# 4. Submit TTS job
job_tts_req = urllib.request.Request(
    f"{base}/jobs",
    data=json.dumps({
        "task": "tts",
        "params": {
            "prompt_audio_asset_id": asset_tts["id"],
            "prompt_text": "同时，他强调微调要科学有序。",
            "target_text": "RTF 实时率因子展示测试。",
            "n_timesteps": 4,
            "max_new_audio_steps": 30,
        },
        "preset": "fast"
    }).encode(),
    headers={"Content-Type": "application/json"}
)
job_tts = json.loads(urllib.request.urlopen(job_tts_req).read().decode())
print(f"Submitted TTS Job #{job_tts['id']}, waiting...")

while True:
    time.sleep(1.0)
    j = json.loads(urllib.request.urlopen(f"{base}/jobs/{job_tts['id']}").read().decode())
    if j["status"] == "completed":
        print(f"✓ TTS Completed! Output duration: {j['result'].get('duration_s')}s | Latency: {j['latency_seconds']}s | RTF: {j['result'].get('rtf')} | Step: {j['current_step']}")
        assert "rtf" in j["result"], "RTF missing in TTS result"
        break
    elif j["status"] == "failed":
        raise AssertionError(f"TTS Job failed: {j['error']}")

print("ALL RTF VERIFICATIONS PASSED!")
