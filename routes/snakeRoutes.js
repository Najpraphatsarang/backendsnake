import express from 'express';
import Grid from 'gridfs-stream';
import mongoose from 'mongoose';
import Snake from '../models/Snake.js';

const router = express.Router();

// ดึงข้อมูลทั้งหมด
router.get('/', async (req, res) => {
    try {
        const snakes = await Snake.find();
        console.log("🐍 Fetched Snakes:", snakes); 
        res.json(snakes);
    } catch (error) {
        console.error("❌ Error fetching snakes:", error);
        res.status(500).json({ message: 'Server Error', error });
    }
});

// ดึงข้อมูลตาม ID
router.get('/:id', async (req, res) => {
    try {
        const snake = await Snake.findById(req.params.id);
        if (!snake) {
            return res.status(404).json({ message: 'Snake not found' });
        }
        res.json(snake);
    } catch (error) {
        res.status(500).json({ message: 'Server Error', error });
    }
});

// ดึงข้อมูลตามชื่อสายพันธุ์ (Binomial)
router.get('/species/:binomial', async (req, res) => {
    try {
        const snake = await Snake.findOne({ binomial: req.params.binomial });

        // หากไม่พบสายพันธุ์ที่ค้นหา
        if (!snake) {
            return res.status(404).json({ message: 'Snake species not found' });
        }

        res.json(snake);  // ส่งข้อมูลสายพันธุ์ที่พบ
    } catch (error) {
        console.error("❌ Error fetching snake species:", error);
        res.status(500).json({ message: 'Server Error', error });
    }
});

// ดึงไฟล์ตามชื่อไฟล์ใน GridFS
router.get('/files/:filename', (req, res) => {
    const filename = req.params.filename;
    const gfs = Grid(mongoose.connection.db, mongoose.mongo);
    gfs.collection('photos'); // หมายถึง collection ที่เก็บไฟล์ใน MongoDB

    gfs.files.findOne({ filename }, (err, file) => {
        if (err || !file) {
            return res.status(404).send('File not found');
        }

        // กำหนด content-type ตามประเภทไฟล์ที่เราเก็บ
        res.set('Content-Type', file.contentType);  // กำหนดประเภทไฟล์ เช่น image/jpeg, image/png

        const readStream = gfs.createReadStream(file.filename); // อ่านไฟล์จาก GridFS
        readStream.pipe(res);  // ส่งไฟล์ไปยัง response และให้ Postman แสดงภาพ
    });
});

export default router;
