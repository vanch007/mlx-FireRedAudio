import urllib.request
import json

base = "http://127.0.0.1:7860/api/v1"

# 1. Model status alias
res1 = json.loads(urllib.request.urlopen(f"{base}/model/status").read().decode())
print("✓ GET /model/status:", res1["status"])

# 2. List jobs & Results endpoint
jobs = json.loads(urllib.request.urlopen(f"{base}/jobs").read().decode())
if jobs:
    jid = jobs[0]["id"]
    res2 = json.loads(urllib.request.urlopen(f"{base}/results/{jid}").read().decode())
    print(f"✓ GET /results/{jid}:", res2["status"])

# 3. Create project & PATCH update
req3 = urllib.request.Request(f"{base}/projects", data=json.dumps({"name": "原始名", "description": "原始说明"}).encode(), headers={"Content-Type": "application/json"})
proj = json.loads(urllib.request.urlopen(req3).read().decode())
req3_patch = urllib.request.Request(f"{base}/projects/{proj['id']}", data=json.dumps({"name": "修改后项目名", "description": "修改后描述"}).encode(), headers={"Content-Type": "application/json"}, method="PATCH")
proj_updated = json.loads(urllib.request.urlopen(req3_patch).read().decode())
print("✓ PATCH /projects/{id}:", proj_updated["name"])

# 4. Media by Asset ID
assets = json.loads(urllib.request.urlopen(f"{base}/assets").read().decode())
if assets:
    aid = assets[0]["id"]
    media_res = urllib.request.urlopen(f"{base}/media/{aid}")
    print(f"✓ GET /media/{aid}: status {media_res.status}, size: {len(media_res.read())} bytes")

print("ALL ENDPOINT CHECKS PASSED!")
