import urllib.request
import json
import time

base = "http://127.0.0.1:7860/api/v1"

job_vd_req = urllib.request.Request(
    f"{base}/jobs",
    data=json.dumps({
        "task": "voice_design",
        "params": {
            "instruction": "温柔知性的广播女主播声音，语调平稳清晰",
            "text": "欢迎收听今日科技前沿播报，音色设计功能现已完美运行。",
            "n_timesteps": 4,
        },
        "preset": "fast"
    }).encode(),
    headers={"Content-Type": "application/json"}
)
job_vd = json.loads(urllib.request.urlopen(job_vd_req).read().decode())
print(f"Submitted Voice Design Job #{job_vd['id']}, polling for completion...")

t0 = time.time()
while time.time() - t0 < 60:
    time.sleep(1.0)
    j = json.loads(urllib.request.urlopen(f"{base}/jobs/{job_vd['id']}").read().decode())
    if j["status"] == "completed":
        print(f"✓ Voice Design SUCCESS! Output: {j['result']['media_url']} (Duration: {j['result']['duration_s']}s, Latency: {j['latency_seconds']}s, RTF: {j['result'].get('rtf')})")
        print(f"  Timbre Description: {j['result'].get('timbre_description')}")
        assert j['result']['duration_s'] > 0, "Audio duration must be positive"
        break
    elif j["status"] == "failed":
        raise AssertionError(f"Voice Design failed: {j['error']}")
    else:
        print(f"  ...status: {j['status']} ({j['current_step']})")

print("VOICE DESIGN LIVE VERIFICATION PASSED!")
