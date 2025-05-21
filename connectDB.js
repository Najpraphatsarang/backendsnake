import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config(); // ✅ โหลดค่า ENV ก่อนใช้งาน!

const connectDB = async () => {
    try {
        console.log(process.env.DATABASE_URL); 
        const mongoURI = process.env.DATABASE_URL;
        if (!mongoURI) throw new Error("DATABASE_URL is missing!");

        const connectionInstance = await mongoose.connect(mongoURI);
        console.log(`✅ MongoDB connected! DB HOST: ${connectionInstance.connection.host}`);
    } catch (error) {
        console.error('❌ MONGO connection error:', error.message);
        process.exit(1);
    }
};

export default connectDB;
