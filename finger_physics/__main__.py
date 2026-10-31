"""Command-line entry point: `python -m finger_physics` or `finger-physics`."""

import argparse


def main():
    ap = argparse.ArgumentParser(
        prog="finger-physics",
        description="Real-time camera finger-joint tracking with a physics-simulated hand skeleton.",
    )
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--video", help="use a video file instead of the camera")
    ap.add_argument("--width", type=int, default=640, help="capture width (default 640)")
    ap.add_argument("--height", type=int, default=480, help="capture height (default 480)")
    ap.add_argument("--fps", type=int, default=60, help="requested camera fps (default 60)")
    ap.add_argument("--scale", type=float, default=1.5, help="window size / capture size (default 1.5)")
    ap.add_argument("--hands", type=int, default=2, choices=[1, 2], help="max hands (default 2)")
    ap.add_argument("--max-fps", type=float, default=120, help="render rate cap (default 120)")
    ap.add_argument("--cpu", action="store_true", help="force the CPU inference delegate")
    ap.add_argument("--selftest", type=float, metavar="SECONDS", help="headless smoke test, no camera")
    ap.add_argument("--screenshot", help="with --selftest: save the last frame here")
    args = ap.parse_args()

    from .app import App  # heavy imports after arg parsing so --help is instant

    App(args).run()


if __name__ == "__main__":
    main()
