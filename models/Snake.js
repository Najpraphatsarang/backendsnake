import mongoose from 'mongoose';

const snakeSchema = new mongoose.Schema({
  binomial: { type: String, required: true },
  poisonous: { type: String, required: true },
  imageUrl: { type: String, required: true },
});

const Snake = mongoose.model('Snake', snakeSchema, 'snake');
export default Snake;
