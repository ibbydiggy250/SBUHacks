🤟 ASL-Zoom-Translator

Bridging the gap between the Deaf community and hearing participants — one sign at a time.

⸻

🧠 Overview

ASL-Zoom-Translator is a real-time American Sign Language (ASL) to English translator designed to integrate seamlessly with Zoom.
The system captures live video, detects hand gestures using MediaPipe, classifies them with a PointNet-based neural network, and sends translated English captions directly into Zoom via the Zoom SDK/API.

Our goal is to make digital communication inclusive by enabling instant ASL translation for virtual meetings.


⸻

🎯 Key Features
-	🖐️ Live Hand Tracking – Uses MediaPipe Hands to extract 3D landmarks from webcam or Zoom feed.

-	🧩 Deep Learning Recognition – Classifies gestures using a lightweight PointNet model trained on ASL data.
-		💬 Real-time Translation – Converts ASL gestures into English words and sentences.

-   Zoom Integration – Injects translated captions or messages into Zoom using the Meeting SDK or Live Transcript API.

-	 🗣️ Optional Speech Output – Speaks out the translated English text for hybrid accessibility.

-	 📊 Fast & Lightweight – Runs smoothly on most laptops; no external servers required.
