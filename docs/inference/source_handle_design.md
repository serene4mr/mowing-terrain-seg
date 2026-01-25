Here is a summary of the sources that the `InferenceSource` class will support. 

This design allows you to use the same command-line argument (`--input`) for everything from a single photo to a live robot camera.

### 1. Single Image
*   **Input Example:** `data/test_grass.jpg`
*   **Identification:** File exists and has a standard image extension (`.jpg`, `.png`, `.jpeg`, `.bmp`).
*   **Behavior:** Yields exactly one frame and then stops.
*   **Use Case:** Quick quality checks on specific challenging images.

### 2. Directory of Images
*   **Input Example:** `data/field_test_v1/`
*   **Identification:** Input is a valid directory path.
*   **Behavior:** Automatically finds all images inside, sorts them alphabetically, and yields them one by one (or in batches if `batch_size > 1`).
*   **Use Case:** Batch processing of a dataset or "frame-by-frame" analysis of a recorded session.

### 3. Video File
*   **Input Example:** `data/mower_run_01.mp4`
*   **Identification:** File exists and has a video extension (`.mp4`, `.avi`, `.mkv`, `.mov`).
*   **Behavior:** Uses OpenCV to extract frames sequentially. It also provides metadata like the video's native FPS so you can save the results at the same speed.
*   **Use Case:** Analyzing pre-recorded field tests.

### 4. Local Camera (Webcam/USB)
*   **Input Example:** `0` or `1`
*   **Identification:** Input is a string that can be converted to an integer.
*   **Behavior:** Opens the local hardware camera. This is an **infinite source** (it runs until you stop it). 
*   **Use Case:** Testing the model "live" on your laptop or directly on the mower hardware.

### 5. Network Stream (RTSP/HTTP)
*   **Input Example:** `rtsp://192.168.1.100:554/stream` or `http://...`
*   **Identification:** String starts with a protocol prefix like `rtsp://`, `http://`, or `https://`.
*   **Behavior:** Connects to a remote camera feed. Like a local camera, this is an **infinite source**.
*   **Use Case:** Remote monitoring of a mower's feed over a wireless network.

---

### Source Capabilities Summary

| Source Type | Data Type | Order | Finite/Infinite |
| :--- | :--- | :--- | :--- |
| **Image** | File Path | N/A | Finite (1) |
| **Directory** | Folder Path | Alphabetical | Finite (N) |
| **Video** | File Path | Sequential | Finite (N) |
| **Camera** | Integer ID | Real-time | **Infinite** |
| **Stream** | URL String | Real-time | **Infinite** |

### Why this set?
This coverage ensures that your `inference.py` tool can handle the entire lifecycle of your project: from **research** (images) to **validation** (videos) to **real-world deployment** (cameras and streams).

---


The `InferenceSource` class acts as the **Data Engine** for your inference system. Its main job is to take any input (image, folder, video, or camera) and turn it into a standardized stream of data for your model.

Here is a summary of its core architecture and responsibilities:

### 1. Initialization (`__init__`)
*   Takes the raw input (`src`) and the desired `batch_size`.
*   Uses a "Brain" (`_detect_type`) to automatically classify the input into an Enum: `IMAGE_FILE`, `IMAGE_DIR`, `VIDEO_FILE`, `CAMERA_ID`, or `STREAM_URL`.

### 2. The Setup Phase (`_setup_source`)
*   **Called by `__iter__`** when the loop starts.
*   **For Folders:** Finds all image files and creates a sorted list.
*   **For Videos/Cameras:** Opens the connection using `cv2.VideoCapture`.
*   **Calculates `total_count`**: Tells the system how many frames to expect (or if it's infinite).

### 3. The Extraction Phase (`__next__`)
*   **The Heart of the loop:**
    *   Reads the next frame(s) from disk or the camera buffer.
    *   Handles **Batching**: Collects $N$ images before returning.
    *   Handles **Metadata**: Attaches the original filename, frame index, and timestamp to every frame.
    *   **Ends the loop**: Raises `StopIteration` when there is no more data.

### 4. Resource Management (The "Cleanup")
*   Implements the **Context Manager** protocol (`__enter__` and `__exit__`).
*   Ensures that when you stop the script, the webcam is released and the video file is closed correctly using a `close()` method.

---

### Why this design is powerful for your Mower project:

| Feature | Benefit |
| :--- | :--- |
| **Auto-Detection** | You can test on a single photo or a live camera using the same script. |
| **Standardized Metadata** | Makes it easy to save results like `frame_0001_mask.png` automatically. |
| **Iterator Pattern** | Keeps memory usage low, even if you are processing a folder with 100,000 images. |
| **Resource Safety** | Prevents the robot's camera from getting "locked" if the code crashes. |

**In short:** This class hides all the "dirty" logic of file paths and video buffers, so your main `inference.py` script can stay clean, readable, and focused on running the model. 
