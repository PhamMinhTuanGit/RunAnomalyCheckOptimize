from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil
import pandas as pd
app = FastAPI()

# Đường dẫn đích cố định
DESTINATION_DIR = "/data/processed"

# Schema cho đầu vào
class FilePath(BaseModel):
    source_path: str

@app.post("/move-file")
def move_file(payload: FilePath):
    source = payload.source_path
    filename = os.path.basename(source)
    destination = os.path.join(DESTINATION_DIR, filename)

    # Kiểm tra file tồn tại
    if not os.path.exists(source):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        shutil.move(source, destination)  # hoặc shutil.copy nếu muốn copy thay vì move
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error moving file: {str(e)}")

    return {
        "message": "File moved successfully",
        "from": source,
        "to": destination
    }
class DataRow(BaseModel):
    id: int
    name: str
    value: float

@app.post("/add-row")
def add_row(row: DataRow):
    global df_global

    # Convert Pydantic model to DataFrame and append
    new_row = pd.DataFrame([row.dict()])
    df_global = pd.concat([df_global, new_row], ignore_index=True)

    return {
        "message": "Row added successfully",
        "current_rows": len(df_global)
    }