## Face Locking Feature

The system allows manual selection of a target identity.
When the selected face is confidently recognized, the system locks onto it.

Once locked:
- the system tracks the same face across frames
- other faces are ignored
- brief recognition failures are tolerated
- the lock is released only after prolonged disappearance

### Detected Actions
While locked, the following actions are detected:
- Face moved left
- Face moved right
- Eye blink
- Smile / laugh (ratio-based)

### Action History
All detected actions are recorded to a file:

data/history/<identity>_history_<timestamp>.txt

Each record contains:
- timestamp
- action type
- value or description
