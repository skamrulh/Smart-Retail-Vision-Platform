"""
Camera consumer: reads frames (webcam or video), pushes frames to Redis stream.
"""
import os
import cv2
import base64
import json
import time
import asyncio
from .redis_client import RedisClient

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CAM_SOURCE = os.getenv("CAM_SOURCE", "0")
SLEEP = float(os.getenv("CAM_SLEEP", "0.1"))

redis = RedisClient(REDIS_URL)

async def produce_from_video(video_source=0, sleep=0.1):
    source = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"WARNING: camera source {video_source} not available. Exiting.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(1)
            continue
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        event = {"frame_b64": frame_b64, "timestamp": time.time()}
        try:
            await redis.xadd("camera:frames", {"data": json.dumps(event)})
        except Exception as e:
            print("Failed to push to redis stream:", e)
        await asyncio.sleep(sleep)

if __name__ == "__main__":
    try:
        asyncio.run(produce_from_video(CAM_SOURCE, SLEEP))
    except KeyboardInterrupt:
        print("Stopping camera producer")
