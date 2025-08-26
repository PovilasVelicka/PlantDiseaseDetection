
from pathlib import Path
import subprocess
import webbrowser
import time

# Define paths
api_dir = Path("plant_api")
frontend_path = Path("plant_frontend/index.html")
onnx_path = api_dir / "plant_disease_model.onnx"
json_path = api_dir / "class_names.json"

# Check required files
if not onnx_path.exists() or not json_path.exists():
    print("❌ Required model files not found in plant_api/. Please copy plant_disease_model.onnx and class_names.json")
else:
    # Start API server using uvicorn
    print("🚀 Starting FastAPI server...")
    api_process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=api_dir,
    )

    # Wait a moment to let server start
    time.sleep(2)

    # Open frontend in browser
    if frontend_path.exists():
        webbrowser.open_new_tab(str(frontend_path.resolve().as_uri()))
        print("🌐 Frontend opened in browser.")
    else:
        print("⚠️ Frontend not found. Make sure plant_frontend/index.html exists.")

    print("✅ API started at http://localhost:8000")
    input("🔴 Press ENTER to stop server...")

    # Stop API server
    api_process.terminate()
    api_process.wait()
    print("🛑 Server stopped.")

