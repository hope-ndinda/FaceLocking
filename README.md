# Face Locking Feature

This project implements a **Face Locking** mechanism on top of an existing
Face Recognition system using **ArcFace ONNX** and **5-Point Face Alignment**.

The system moves beyond simple recognition by locking onto a selected identity
and tracking that person’s behavior over time.

---

## Manual Face Selection

The target identity is manually selected in the code:

```python
TARGET_IDENTITY = "Hope"

Only this enrolled identity can be locked and tracked.
Other detected faces are ignored once the lock is active.

How Face Locking Works

Multiple faces are detected in each camera frame.

Each face is aligned using 5 facial landmarks.

ArcFace embeddings are extracted and compared with the enrolled database.

When the selected identity is confidently recognized:

the system locks onto that face

a clear LOCKED indicator is displayed

While locked:

the same face is tracked across frames

other faces are ignored

brief recognition failures are tolerated

The lock is released only if the face disappears for a prolonged time.

## Detected Actions

While the face is locked, the system detects and records the following actions
using simple, explainable geometric rules:

-Face moved left
-Face moved right
-Eye blink
-Smile / laugh (ratio-based detection)


### Action History Recording

All detected actions are recorded to a history file while the face is locked.

File naming format:

data/history/<identity>_history_<timestamp>.txt


Each record includes:

-timestamp
-action type
-brief description or value

# How to Run

Activate the virtual environment and run:

python -m src.recognize


# Notes on Repository Contents

For size and privacy reasons, the following files are excluded from this repository:

-ONNX model files
-Face embedding databases
-Generated action history files
These files are expected to be available locally when running the system.