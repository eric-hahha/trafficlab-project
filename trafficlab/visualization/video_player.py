import cv2


class VideoPlayer:
    """Backend video playback helper wrapping cv2.VideoCapture.

    Encapsulates all frame-seeking and frame-reading operations so that
    GUI code does not need to call cv2 directly.
    """

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def seek(self, index):
        """Seek to a specific frame index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def read(self):
        """Read the next frame. Returns (ret, frame) like cv2.VideoCapture.read()."""
        return self.cap.read()

    def read_frame(self, index=None):
        """Optionally seek to *index*, then read and return the frame.

        Returns the frame (numpy array) on success, or None on failure.
        """
        if index is not None:
            self.seek(index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def frame_count(self):
        """Return the total number of frames reported by the container."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def fps(self):
        """Return the nominal frames-per-second of the video."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def resolution(self):
        """Return (width, height) in pixels."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        """Release the underlying VideoCapture resource."""
        if self.cap:
            self.cap.release()
