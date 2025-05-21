from itsdangerous import URLSafeTimedSerializer
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
import aiosmtplib
from email.message import EmailMessage

# Load environment variables
load_dotenv()

# Initialize serializer with secret key
SECRET_KEY = os.getenv("SECRET_KEY")
serializer = URLSafeTimedSerializer(SECRET_KEY)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Generate token for password reset
def generate_reset_token(email: str):
    return serializer.dumps(email, salt="reset-password")

# Verify the reset token
def verify_reset_token(token: str, expiration=3600):
    try:
        email = serializer.loads(token, salt="reset-password", max_age=expiration)
        return email
    except Exception:
        return None

# Hash password using bcrypt
def hash_password(password: str):
    return pwd_context.hash(password)

async def send_reset_email(to_email: str, token: str):
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"


    # สร้างข้อความ HTML
    html_content = f"""
    <html>
        <body>
            <h2>รีเซ็ตรหัสผ่านของคุณ</h2>
            <p>คุณได้รับอีเมลนี้เพราะมีการร้องขอให้เปลี่ยนรหัสผ่านสำหรับบัญชีของคุณ</p>
            <p>หากคุณต้องการเปลี่ยนรหัสผ่าน กรุณาคลิกลิงก์ด้านล่าง:</p>
            <p>
                <a href="{reset_link}" style="color:blue;text-decoration:underline;">
                    คลิกที่นี่เพื่อรีเซ็ตรหัสผ่าน
                </a>
            </p>
            <p>หากคุณไม่ได้ร้องขอการเปลี่ยนรหัสผ่าน กรุณาละเว้นอีเมลนี้</p>
            <br>
            <p>ขอบคุณครับ,</p>
            <p><strong>ทีมงานของเรา</strong></p>
        </body>
    </html>
    """

    # สร้าง EmailMessage
    message = EmailMessage()
    message["From"] = os.getenv("EMAIL_USERNAME")
    message["To"] = to_email
    message["Subject"] = "Reset your password"

    # ต้องใส่ plain text ก่อน แล้วค่อย add_alternative HTML
    message.set_content(f"คุณสามารถรีเซ็ตรหัสผ่านได้ที่ลิงก์นี้: {reset_link}")
    message.add_alternative(html_content, subtype="html")

    # ส่งอีเมล
    await aiosmtplib.send(
        message,
        hostname=os.getenv("EMAIL_HOST"),
        port=int(os.getenv("EMAIL_PORT")),
        start_tls=True,
        username=os.getenv("EMAIL_USERNAME"),
        password=os.getenv("EMAIL_PASSWORD"),
    )
