"""
Video Export — Renders simulation frames as screenshots and stitches into MP4.

Uses Playwright (headless Chromium) to screenshot each folium map frame,
then ffmpeg to encode into a compact MP4 video.

Usage:
    from video_export import export_simulation_video
    export_simulation_video(frames_html, output_path="simulation_video.mp4", fps=0.4)
"""
import os
import subprocess
import tempfile
import shutil


def export_simulation_video(frames_html: list, output_path: str = "simulation_video.mp4",
                            seconds_per_frame: float = 2.5, width: int = 1280, height: int = 800):
    """
    Render simulation frames to video.

    Args:
        frames_html: List of HTML strings (one per frame, from folium maps)
        output_path: Where to save the MP4 file
        seconds_per_frame: How long each frame shows (2.5s = 60s video for 24 frames)
        width: Video width in pixels
        height: Video height in pixels

    Returns:
        Path to the saved video file, or None on failure
    """
    from playwright.sync_api import sync_playwright

    if not frames_html:
        print("[VideoExport] No frames to export")
        return None

    # Create temp directory for frame images
    tmp_dir = tempfile.mkdtemp(prefix="sim_frames_")
    print(f"[VideoExport] Rendering {len(frames_html)} frames at {width}x{height}...")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})

            for i, frame_html in enumerate(frames_html):
                # Write frame HTML to temp file
                frame_file = os.path.join(tmp_dir, f"frame_{i:03d}.html")
                with open(frame_file, "w", encoding="utf-8") as f:
                    f.write(frame_html)

                # Navigate and wait for map tiles to load
                page.goto(f"file://{frame_file}")
                page.wait_for_timeout(2000)  # Wait 2s for tiles to render

                # Screenshot
                screenshot_path = os.path.join(tmp_dir, f"frame_{i:03d}.png")
                page.screenshot(path=screenshot_path)

                print(f"  Frame {i+1}/{len(frames_html)} rendered", end="\r")

            browser.close()

        print(f"\n[VideoExport] All frames rendered. Encoding video...")

        # Use ffmpeg to stitch frames into MP4
        fps = 1.0 / seconds_per_frame
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%03d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            "-vf", f"scale={width}:{height}",
            output_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[VideoExport] ffmpeg error: {result.stderr[:500]}")
            return None

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        duration = len(frames_html) * seconds_per_frame
        print(f"[VideoExport] ✅ Video saved: {output_path} ({file_size:.1f} MB, {duration:.0f}s)")
        return output_path

    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Test with a simple frame
    test_frames = [
        "<html><body style='background:#1a1a1a;display:flex;align-items:center;justify-content:center;height:100vh;'><h1 style='color:white;font-family:monospace;'>Frame 1 — H+00</h1></body></html>",
        "<html><body style='background:#1a1a1a;display:flex;align-items:center;justify-content:center;height:100vh;'><h1 style='color:white;font-family:monospace;'>Frame 2 — H+02</h1></body></html>",
        "<html><body style='background:#1a1a1a;display:flex;align-items:center;justify-content:center;height:100vh;'><h1 style='color:white;font-family:monospace;'>Frame 3 — H+04</h1></body></html>",
    ]
    export_simulation_video(test_frames, "test_video.mp4", seconds_per_frame=2.0)
