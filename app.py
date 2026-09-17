from __future__ import annotations
import json
import os
import subprocess
import tarfile
import time
import urllib.request
import zipfile

from fastapi import Request
from fastapi.responses import StreamingResponse
from gradio import Server
import httpx

app = Server()

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

@spaces.GPU if HAS_SPACES else lambda f: f
def whygpu():
    pass

def build_latest_llama_server():
    binary_path = os.path.abspath("llama.cpp/build/bin/llama-server")
    if os.path.exists(binary_path):
        return binary_path

    print("Building latest llama.cpp from master source...")

    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git"], check=True)
    else:
        subprocess.run(["git", "pull"], cwd="llama.cpp", check=True)

    env = os.environ.copy()
    if "/usr/local/cuda/bin" not in env.get("PATH", ""):
        env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"

    cmake_cmd = [
        "cmake",
        "-B",
        "build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=ON",
        "-DGGML_OPENMP=ON",
        "-DGGML_AVX512=ON",
    ]

    subprocess.run(cmake_cmd, cwd="llama.cpp", env=env, check=True)
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 4), "--target", "llama-server"],
        cwd="llama.cpp",
        env=env,
        check=True
    )

    if not os.path.exists(binary_path):
        raise RuntimeError("Failed to build llama-server binary!")

    return binary_path

def apathy_exe():
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen3.8-27B",
        filename="q.gguf",
    )

def start_llama_server():
    binary_path = build_latest_llama_server()

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-UD-Q8_K_XL.gguf",
    )

    draft_path = hf_hub_download(
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename="MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
    )

    binary_dir = os.path.dirname(os.path.abspath(binary_path))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{binary_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        binary_path,
        "-m", model_path,
        "-md", draft_path,
        "--port", "8000",
        "--host", "127.0.0.1",
        "-t", "16",
        "-tb", "16",
        "-fa", "on",
        "--parallel", "1",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "20",
        "--spec-draft-p-min", "0.70",
        "--spec-draft-n-min", "1",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--jinja",
        "--temp", "1.0",
        "--top-p", "0.95",
        "--top-k", "20",
        "--min-p", "0.0",
        "--presence-penalty", "0.0",
        "--repeat-penalty", "1.0",
    ]

    process = subprocess.Popen(cmd, env=env)
    time.sleep(5)
    return process

@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/gradio_api"):
        return await call_next(request)

    url = f"http://127.0.0.1:8000{path}"
    
    headers = {
        k: v for k, v in request.headers.items() 
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }
    body = await request.body()

    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=120.0))
    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        params=request.query_params,
        content=body,
    )

    try:
        response = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        return StreamingResponse(iter([f"Proxy Error: {e}".encode()]), status_code=502)

    async def stream_and_close():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    res_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in ("content-length", "transfer-encoding", "connection")
    }

    return StreamingResponse(
        stream_and_close(),
        status_code=response.status_code,
        headers=res_headers
    )

if __name__ == "__main__":
    start_llama_server()
    app.launch(show_error=True)