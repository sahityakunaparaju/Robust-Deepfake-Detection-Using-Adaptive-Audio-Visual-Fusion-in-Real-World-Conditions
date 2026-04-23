from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
import time
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from model import analyze_video

app = FastAPI(title="Explainable Deepfake Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "frames"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/api/analyze")
async def analyze_endpoint(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(('.mp4', '.avi')):
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Only .mp4 and .avi are supported."})
    
    file_path = os.path.join(UPLOAD_DIR, video.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        start_time = time.time()
        
        result = analyze_video(file_path)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        
        processing_time = round(time.time() - start_time, 2)
        result["processing_time"] = f"{processing_time}s"
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An internal server error occurred: {str(e)}"})
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.post("/api/report")
async def generate_report(data: dict = Body(...)):
    try:
        temp_pdf = tempfile.mktemp(suffix='.pdf')
        c = canvas.Canvas(temp_pdf, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 50, "Explainable Deepfake Analysis Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 90, f"Prediction: {data.get('prediction', 'Unknown')}")
        c.drawString(50, height - 110, f"Confidence: {int(data.get('confidence', 0) * 100)}%")
        c.drawString(50, height - 130, f"Stability Score: {data.get('stability_score', 0)}")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 170, "Decision Summary")
        c.setFont("Helvetica", 10)
        
        textobject = c.beginText(50, height - 190)
        summary = data.get('decision_summary', '')
        # Simple text wrapping abstraction
        max_len = 90
        lines = [summary[i:i+max_len] for i in range(0, len(summary), max_len)]
        for line in lines:
            textobject.textLine(line)
        c.drawText(textobject)
        
        y_pos = height - 260
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Fusion Weights")
        c.setFont("Helvetica", 12)
        fw = data.get('fusion_weights', {})
        c.drawString(50, y_pos - 20, f"Visual Weight: {fw.get('video', 0)}")
        c.drawString(50, y_pos - 40, f"Audio Weight: {fw.get('audio', 0)}")
        
        y_pos -= 80
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Top Manipulated Frames")
        y_pos -= 20
        c.setFont("Helvetica", 10)
        
        top_frames = data.get('top_frames', [])
        for f in top_frames:
            c.drawString(50, y_pos, f"Frame {f.get('frame_index')}: Score {f.get('score')}")
            y_pos -= 15
            
        c.showPage()
        c.save()
        
        return FileResponse(temp_pdf, media_type='application/pdf', filename='analysis_report.pdf')
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate PDF: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
