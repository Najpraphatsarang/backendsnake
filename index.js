import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import snakeRoutes from './routes/snakeRoutes.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// ✅ เปิดใช้งาน CORS
app.use(cors({
    origin: 'http://localhost:3000', // ✅ ให้ frontend เรียก backend ได้
    credentials: true
}));

// ✅ เชื่อม MongoDB
const connectDB = async () => {
    try {
        await mongoose.connect(process.env.DATABASE_URL);
        console.log("✅ MongoDB connected!");
    } catch (error) {
        console.log("❌ MONGO connection error:", error);
        process.exit(1);
    }
};
connectDB();

// ✅ Middleware
app.use(express.json());
app.use('/api/snakes', snakeRoutes);

// ✅ Start Server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});
