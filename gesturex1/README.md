# GestureX

Sign language to text and speech, with a video call where each person's signs are spoken to the other.

## Run it

The camera only works on `https://` or `http://localhost`, so serve the folder instead of opening the file directly:

```
cd gesturex
python3 -m http.server 8000
```

Open http://localhost:8000 in Chrome or Edge. To use it across two devices, deploy the folder to any static host (Netlify, Vercel, GitHub Pages, Cloudflare Pages), which gives you https.

## How it works

- **Hand tracking**: MediaPipe Hand Landmarker (runs in the browser, nothing is uploaded).
- **Recognition**: the 21 hand points are normalised and matched against a dataset with k-nearest-neighbours. If nothing in the dataset is close, built-in rules recognise a few signs: Hello, Yes, Stop, Peace, Good, OK, I love you, Water, Look, I.
- **Speech**: the browser's Web Speech API.
- **Video call**: WebRTC through PeerJS. Each person's browser recognises their own signs and sends the word over the call's data channel, so the other person sees it as a caption and hears it.

## Using an existing dataset

1. Download an image dataset organised as one folder per sign (for example the Kaggle "ASL Alphabet" dataset).
2. Install the tools and run the converter:
   ```
   pip install mediapipe opencv-python
   python build_dataset.py path/to/dataset --per-class 300
   ```
   If your photos are selfie-mirrored and results look swapped, add `--flip`.
3. This writes `model/landmarks.json`. Reload the app; the Dataset panel shows how many samples loaded.

You can also teach signs inside the app (Recognize tab, Teach a sign). They are stored in your browser.

## Limits

- Letters and a handful of words from still hand shapes. Signs that depend on motion (most real ASL words) need a sequence model trained on a video dataset such as WLASL; that is a larger step.
- Accounts are stored in this browser only. Google sign-in and password reset need a real backend.
- The call uses PeerJS's free public signalling server and public STUN servers. For production reliability across strict networks, run your own PeerJS server and add a TURN server.
