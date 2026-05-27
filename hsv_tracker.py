import cv2
import numpy as np
import os
import json
import argparse


def nothing(x):
    """Callback function for OpenCV trackbars."""
    pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive HSV threshold tracker for PCB trace extraction."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join("PCB_Dataset", "04.JPG"),
        help="Path to input PCB image."
    )

    parser.add_argument(
        "--output_config",
        type=str,
        default=os.path.join("configs", "hsv_trace_config.json"),
        help="Path to save HSV threshold config."
    )

    parser.add_argument(
        "--display_width",
        type=int,
        default=1000,
        help="Display width for preview window."
    )

    return parser.parse_args()


def resize_keep_aspect(img, target_width: int):
    h, w = img.shape[:2]
    scale = target_width / float(w)
    target_height = int(h * scale)
    resized = cv2.resize(img, (target_width, target_height))
    return resized


def main():
    args = parse_args()

    img = cv2.imread(args.image)

    if img is None:
        print(f"[ERROR] Could not load image from: {args.image}")
        return

    display_img = resize_keep_aspect(img, args.display_width)

    window_name = "HSV Threshold Tracker - Press S to save, Q/ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("H_Min", window_name, 0, 179, nothing)
    cv2.createTrackbar("S_Min", window_name, 0, 255, nothing)
    cv2.createTrackbar("V_Min", window_name, 0, 255, nothing)

    cv2.createTrackbar("H_Max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S_Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V_Max", window_name, 255, 255, nothing)

    # Optional: initialize with your current useful range
    cv2.setTrackbarPos("H_Min", window_name, 35)
    cv2.setTrackbarPos("S_Min", window_name, 0)
    cv2.setTrackbarPos("V_Min", window_name, 0)

    cv2.setTrackbarPos("H_Max", window_name, 90)
    cv2.setTrackbarPos("S_Max", window_name, 255)
    cv2.setTrackbarPos("V_Max", window_name, 75)

    print("[INFO] Starting HSV Tracker...")
    print("[INFO] Adjust the sliders until PCB traces are well isolated.")
    print("[INFO] Press 's' to save current HSV config.")
    print("[INFO] Press 'q' or ESC to quit.\n")

    previous_values = None

    while True:
        h_min = cv2.getTrackbarPos("H_Min", window_name)
        s_min = cv2.getTrackbarPos("S_Min", window_name)
        v_min = cv2.getTrackbarPos("V_Min", window_name)

        h_max = cv2.getTrackbarPos("H_Max", window_name)
        s_max = cv2.getTrackbarPos("S_Max", window_name)
        v_max = cv2.getTrackbarPos("V_Max", window_name)

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        hsv_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        result = cv2.bitwise_and(display_img, display_img, mask=mask)

        # Show original + mask + result together
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        preview = np.hstack([
            display_img,
            mask_bgr,
            result
        ])

        current_values = (h_min, s_min, v_min, h_max, s_max, v_max)

        if current_values != previous_values:
            print(
                f"[UPDATE] Lower: [{h_min}, {s_min}, {v_min}] | "
                f"Upper: [{h_max}, {s_max}, {v_max}]"
            )
            previous_values = current_values

        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            config = {
                "trace_hsv_lower": [int(h_min), int(s_min), int(v_min)],
                "trace_hsv_upper": [int(h_max), int(s_max), int(v_max)]
            }

            output_dir = os.path.dirname(args.output_config)
            ensure_dir(output_dir)

            with open(args.output_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

            print(f"[SUCCESS] HSV config saved to: {args.output_config}")
            print(json.dumps(config, indent=4))

        elif key == 27 or key == ord("q"):
            print("[INFO] Exiting HSV Tracker.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()