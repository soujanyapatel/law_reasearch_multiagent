import subprocess
import time
import webbrowser
import os
import sys

def start_backend():
    print("🚀 Starting Backend Server...")
    backend_process = subprocess.Popen([
        sys.executable, "backend/app.py"
    ], cwd=os.getcwd())
    return backend_process

def start_frontend():
    print("🌐 Starting Frontend Server...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "http.server", "8080"
    ], cwd="frontend")
    return frontend_process

def main():
    print("=" * 50)
    print("🏛️  Law Research Assistant")
    print("=" * 50)
    
    # Check if backend directory exists
    if not os.path.exists("backend/app.py"):
        print("❌ Backend not found! Please ensure backend/app.py exists.")
        return
    
    # Check if frontend directory exists
    if not os.path.exists("frontend/index.html"):
        print("❌ Frontend not found! Please ensure frontend/index.html exists.")
        return
    
    try:
        # Start backend
        backend_proc = start_backend()
        time.sleep(3)  # Wait for backend to start
        
        # Start frontend
        frontend_proc = start_frontend()
        time.sleep(2)  # Wait for frontend to start
        
        print("\n✅ Servers Started Successfully!")
        print("🔗 Backend API: http://localhost:5000")
        print("🔗 Frontend UI: http://localhost:8080")
        print("\n🌐 Opening browser...")
        
        # Open browser
        webbrowser.open("http://localhost:8080")
        
        print("\n📋 Instructions:")
        print("1. Upload a legal paper (PDF/TXT/DOCX)")
        print("2. Choose query type (Find Sources, Summarize, etc.)")
        print("3. Ask questions or analyze arguments")
        print("\n⚠️  Press Ctrl+C twice to stop both servers")
        
        # Keep script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_proc.terminate()
            frontend_proc.terminate()
            print("✅ Servers stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error starting servers: {e}")

if __name__ == "__main__":
    main()