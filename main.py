from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import base64
import json
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn  # ตรวจสอบให้แน่ใจว่าอิมพอร์ต nn ที่นี่
from pydantic import BaseModel, EmailStr
import bcrypt
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from uuid import uuid4
from typing import List, Optional
from model import ForgotPasswordRequest, ResetPasswordRequest
from utils import generate_reset_token, verify_reset_token, send_reset_email, hash_password

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


app = FastAPI()

# ✅ อนุญาตให้ frontend เรียก API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:5173"],  # อนุญาต React ที่รันบน localhost:3000
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Method (GET, POST, PUT, DELETE)
    allow_headers=["*"],
)

# ✅ โหลด ENV ตัวแปร
load_dotenv()

# SECRET_KEY = os.getenv("SECRET_KEY")
# ALGORITHM = os.getenv("ALGORITHM")
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
# print("🔑 SECRET_KEY:", SECRET_KEY)
# print("🔐 ALGORITHM:", ALGORITHM)

# ✅ เชื่อมต่อ MongoDB
# MONGO_URI = "mongodb://localhost:27017"
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# DB_NAME = "snake"
# client = AsyncIOMotorClient(MONGO_URI)
# MONGO_URI = os.getenv("MONGO_URI")
# client = AsyncIOMotorClient(MONGO_URI)
# db = client.get_database() 
# db = client[DB_NAME]
# snake_collection = db["snake"]    
# admin_collection = db["admin"] 

class Snake(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    binomial: str
    thai_name: str
    description: str
    danger_level: str
    is_venomous: bool
    poisonous: Optional[str] = None
    color: List[str]
    diet: List[str]
    habitat: List[str]
    pattern: Optional[str] = None
    size: Optional[str] = None
    venom_effects: Optional[str] = None
    imageUrl: Optional[str] = None

class SnakeUpdate(BaseModel):
    binomial: Optional[str] = None
    thai_name: Optional[str] = None
    description: Optional[str] = None
    danger_level: Optional[str] = None
    is_venomous: Optional[bool] = None
    poisonous: Optional[str] = None
    color: Optional[List[str]] = None
    diet: Optional[List[str]] = None
    habitat: Optional[List[str]] = None
    pattern: Optional[str] = None
    size: Optional[str] = None
    venom_effects: Optional[str] = None
    imageUrl: Optional[str] = None


SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# ✅ เชื่อมต่อ MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_database()  # เข้าถึงฐานข้อมูลโดยใช้ชื่อที่ตั้งใน URI

snake_collection = db["snake"]    
admin_collection = db["admin"]

# ✅ โหลดโมเดล PyTorch
MODEL_PATH = r"C:\Users\SURFACE PRO X SQ1\project\backend\resnet18_trained.pth"

# ✅ สร้างโมเดล ResNet18
model = models.resnet18(weights=None)  # ใช้ weights=None ถ้าคุณไม่ต้องการโหลด weights เริ่มต้น
num_classes = 6  # ปรับจำนวนคลาสให้ตรงกับโมเดล
model.fc = nn.Linear(model.fc.in_features, num_classes)  # ปรับชั้นสุดท้าย

# ✅ โหลดโมเดลที่บันทึกไว้
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"🔥 Error loading model: {e}")

# ✅ กำหนดชื่อคลาสให้ตรงกับโมเดล (ต้องเป็น 5 คลาส)
CLASS_NAMES = [
    'Gonyosoma oxycephalum',
    'Naja atra',
    'Ophiophagus hannah',
    'Psammodynastes pulverulentus',
    'Tropidolaemus wagleri',
    'Xenochrophis piscator'
]

# ✅ ฟังก์ชันแปลงภาพให้เป็น Tensor
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ปรับขนาดให้ตรงกับโมเดล
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # เพิ่ม batch dimension

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/addsnake", response_model=Snake)
async def add_snake(snake: Snake):
    snake_dict = snake.dict()
    result = await snake_collection.insert_one(snake_dict)

    if result.inserted_id:
        return snake
    else:
        raise HTTPException(status_code=500, detail="Failed to insert snake")


@app.put("/snakes/binomial/{binomial}", response_model=Snake)
async def update_snake(binomial: str, updated_data: SnakeUpdate):
    update_fields = {k: v for k, v in updated_data.dict(exclude_unset=True).items() if v is not None}

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    result = await snake_collection.find_one_and_update(
        {"binomial": binomial},
        {"$set": update_fields},
        return_document=True  # ให้คืนค่าหลังอัปเดต
    )

    if result:
        return Snake(**result)
    else:
        raise HTTPException(status_code=404, detail="Snake not found")

@app.delete("/snakes/binomial/{binomial}", response_model=dict)
async def delete_snake(binomial:str):
    result = await snake_collection.delete_one({"binomial":binomial})

    if result.deleted_count == 1:
        return {"message": f"Snake '{binomial}' has been deleted"}
    else:
        raise HTTPException(status_code=404, detail="Snake not found")
@app.post("/login")

async def login_admin(login: LoginRequest):
    print("✅ Login route called")
    print(f"📨 Email: {login.email}")
    print(f"📨 Password: {login.password}")
    try:
        user = await admin_collection.find_one({"email": login.email})
        if not user:
            # return {"error": "❌ ไม่พบผู้ใช้งาน"}
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="❌ ไม่พบผู้ใช้งาน"
            )

        if not bcrypt.checkpw(login.password.encode('utf-8'), user["password"].encode('utf-8')):
            # return {"error": "❌ รหัสผ่านไม่ถูกต้อง"}
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="❌ รหัสผ่านไม่ถูกต้อง"
            )

        # ✅ สร้าง JWT Token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": login.email}, expires_delta=access_token_expires)

        return {
            "message": "✅ เข้าสู่ระบบสำเร็จ",
            "access_token": access_token,
            "token_type": "bearer",
            "email": login.email
        }

    except Exception as e:
         raise e  # ส่ง error code กลับไปยัง client อย่างถูกต้อง
    except Exception as e:
        print(f"🔥 Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❗ เกิดข้อผิดพลาดในระบบ"
        )

# ✅ เช็คว่าเซิร์ฟเวอร์ทำงานอยู่ไหม
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
@app.post("/register")
async def register_admin(admin: RegisterRequest):
    try:
        existing_admin = await admin_collection.find_one({"email": admin.email})
        if existing_admin:
            return {"error": "⚠️ อีเมลนี้ถูกใช้งานแล้ว"}
        hashed_pw = bcrypt.hashpw(admin.password.encode('utf-8'), bcrypt.gensalt())

        # ✅ สร้างข้อมูลผู้ใช้
        user_data = {
            "email": admin.email,
            "password": hashed_pw.decode('utf-8')  # แปลง bytes -> string
        }

        # ✅ บันทึกลง MongoDB
        result = await admin_collection.insert_one(user_data)

        return{
            "message": "✅ สมัครสมาชิกสำเร็จ",
            "user_id": str(result.inserted_id),
            "email": admin.email
        }
    except Exception as e:
        print(f"🔥 Error: {e}")
        return {"error": str(e)}
    
@app.get("/searchsnake")
async def get_snake(search: str = Query(None, description = "ค้นหางู")):
    try:
        query = {}
        if search:
            query = {
                "$or": [
                    {"binomial": {"$regex": search, "$options": "i"}},  # ค้นหาจากชื่อวิทยาศาสตร์
                    {"thai_name": {"$regex": search, "$options": "i"}}  # ค้นหาจากชื่อไทย
                ]
            }

            snakes_cursor = snake_collection.find(query)
            snakes = await snakes_cursor.to_list(length=100)

            if not snakes:
                 return {"message": "❌ ไม่พบข้อมูลงู"}
            
            # แปลง ObjectId เป็น string
            for snake in snakes:
                snake["_id"] = str(snake["_id"])

            return {"snakes": snakes}
    except Exception as e:
        print(f"🔥 Error: {e}")
        return {"error": str(e)}
    
# ✅ ดึงข้อมูลงูจาก MongoDB
@app.get("/snake")
async def get_snake():
    try:
        snakes_cursor = snake_collection.find({})
        snakes = await snakes_cursor.to_list(length=100)

        # Debug: แสดงข้อมูลที่ดึงมาได้
        print(f"🐍 Snakes Query Result: {snakes}")

        if not snakes:
            return {"message": "❌ ไม่พบข้อมูลงูใน database"}

        # แปลง ObjectId เป็น string
        for snake in snakes:
            snake["_id"] = str(snake["_id"])

        return {"snakes": snakes}

    except Exception as e:
        print(f"🔥 Error: {e}")
        return {"error": str(e)}
    
@app.get("/snake_info/{binomial_name}")
async def get_snake_info(binomial_name: str):
    """ ดึงข้อมูลงูจากชื่อสายพันธุ์ """
    snake = await snake_collection.find_one({"binomial": binomial_name})
    if not snake:
        return {"error": "❌ ไม่พบนงูในฐานข้อมูล"}
    
    # แปลง `_id` เป็น string เพื่อให้ JSON ส่งออกได้
    snake["_id"] = str(snake["_id"])
    
    return {"snake": snake}

@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    user = await admin_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not found")
    token = generate_reset_token(request.email)
    await send_reset_email(request.email, token)
    return {"message": "Reset password email sent"}

@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    email = verify_reset_token(request.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    
    hashed = hash_password(request.new_password)
    result = await admin_collection.update_one({"email": email}, {"$set": {"password": hashed}})
    
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Password not updated")
    
    return {"message": "Password updated successfully"}


# ✅ อัพโหลดภาพแล้วทำนายงู
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # อ่านไฟล์รูป
        contents = await file.read()
        print(f"📸 Received file: {file.filename}, Size: {len(contents)} bytes")

        # แปลงภาพเป็น Tensor
        image = Image.open(BytesIO(contents)).convert("RGB")
        img_tensor = transform_image(image)

        # ทำนายผล
        with torch.no_grad():
            predictions = model(img_tensor)
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
            predicted_class = CLASS_NAMES[torch.argmax(probabilities).item()]
            confidence = probabilities.max().item()

        # ดึงข้อมูลงูจาก MongoDB
        snake_data = await snake_collection.find_one({"binomial": predicted_class}, {"_id": 0})

        # แปลงภาพเป็น base64 สำหรับ frontend แสดงผล
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_str}"

        # Response
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "uploaded_image": img_data_url  # << เพิ่มส่วนนี้!
        }

        if snake_data:
            result["snake_info"] = snake_data
        else:
            result["message"] = "⚠️ ไม่พบข้อมูลในฐานข้อมูล"

        return result

    except Exception as e:
        print(f"🔥 Error: {e}")
        return {"error": str(e)}



# ✅ รันเซิร์ฟเวอร์
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
