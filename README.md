Smart Study Posture Alerter
A real-time computer vision application that runs entirely on your laptop using only a webcam. It detects when you slouch while studying or working, and alerts you with a beep after 30 seconds of poor posture.

Features
 Live webcam feed with mirrored view
 Pose estimation using MediaPipe (CPU‑only, no GPU needed)
 Slouch detection based on ear‑to‑shoulder angle relative to vertical
 Persistent slouch timer – only alerts after 30 continuous seconds of slouching
 Audible alert (Windows beep or terminal bell)
 Visual feedback – green “Upright” / red “Slouching” + real‑time angle display

How It Works
MediaPipe Pose extracts 33 body landmarks from each video frame.
The positions of the ears and shoulders are used to compute the angle between the shoulder‑to‑ear vector and the upward vertical.
If the angle exceeds a threshold (default 30°), the posture is classified as slouching.
A state machine tracks how long slouching has been continuous.
After 30 seconds of slouching, an audible alert is triggered.
