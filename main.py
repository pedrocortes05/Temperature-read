import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import csv
import pandas as pd


frame_sample_rate = 1  # in seconds

# === Settings for color bar ===
bar_width_ratio_color = 0.01
bar_width_ratio = 0.08
temp_min = 21
temp_max = 80

def build_color_temperature_map(frame, temp_min, temp_max):
    """Extract the vertical color bar and create a LUT from color to temperature, with debug image output"""
    h, w = frame.shape[:2]
    bar_w = int(w * bar_width_ratio_color)
    bar = frame[:, :bar_w]

    # Sample color along the vertical center of the bar
    sample_x = int(bar_w / 2)
    samples = []
    for y in range(bar.shape[0]):
        color = bar[y, sample_x].astype(np.float32)
        samples.append(color)

    temps = np.linspace(temp_max, temp_min, len(samples))  # Top = hot, bottom = cold

    # === Visualization of the color-temperature map ===
    vis_height = 400
    vis_width = 100
    color_bar_image = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    for i in range(vis_height):
        idx = int(i * len(samples) / vis_height)
        color = samples[idx].astype(np.uint8)
        color_bar_image[i, :] = color

    # Add temperature labels
    vis_image = color_bar_image.copy()
    for i in range(0, vis_height, 40):
        temp_val = temps[int(i * len(temps) / vis_height)]
        cv2.putText(vis_image, f"{int(temp_val)}°C", (5, i + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # cv2.imshow("Extracted Color Bar with Temps", vis_image)
    # cv2.waitKey(0)
    return list(zip(cv2.cvtColor(np.array(samples, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3), temps))


def match_temperature(hsv_pixel, lut):
    """Find the closest HSV color match in the LUT"""
    distances = [np.linalg.norm(hsv_pixel - color) for color, _ in lut]
    min_index = np.argmin(distances)
    return lut[min_index][1]  # Return matched temperature

def process_video(video_path, save_plot=False, save_frames=False, show_plot=False):
    # === Open video and initialize ===
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_sample_rate)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = []
    temperatures = []

    frame_idx = 0
    lut = None
    base_name = os.path.basename(video_path)
    name_no_ext = os.path.splitext(base_name)[0]

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if lut is None:
                lut = build_color_temperature_map(frame, temp_min, temp_max)

            # Crop out the bar to avoid using it in temperature detection
            h, w = frame.shape[:2]
            frame_cropped = frame[:, int(w * bar_width_ratio):]

            # Convert to HSV
            hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)
            # cv2.imshow("Hot Spot Detection", hsv)


            # Reshape HSV image for vectorized matching
            reshaped_hsv = hsv.reshape(-1, 3)
            value_channel = reshaped_hsv[:, 2]

            # Create a mask for non-black pixels (V > 30)
            valid_mask = value_channel > 30
            valid_pixels = reshaped_hsv[valid_mask]

            if valid_pixels.size == 0:
                estimated_temp = temp_min  # fallback if frame is all black
                max_y, max_x = 0, 0
            else:
                # Vectorized distance matching with LUT
                lut_colors = np.array([color for color, _ in lut])
                lut_temps = np.array([temp for _, temp in lut])
                dists = np.linalg.norm(valid_pixels[:, None] - lut_colors[None, :], axis=2)

                best_match_idx = np.argmin(dists, axis=1)
                pixel_temps = lut_temps[best_match_idx]

                max_temp_idx = np.argmax(pixel_temps)
                max_temp = pixel_temps[max_temp_idx]
                estimated_temp = max_temp

                # Map back to 2D coordinates
                valid_indices = np.flatnonzero(valid_mask)
                max_flat_idx = valid_indices[max_temp_idx]
                max_y, max_x = divmod(max_flat_idx, hsv.shape[1])


            # Match it to temperature using LUT
            # estimated_temp = match_temperature(hot_pixel_hsv, lut)
            # print(f"{minVal=}, {maxVal=} {estimated_temp=} {hot_pixel_hsv=}")

            time_sec = frame_idx / fps
            timestamps.append(round(time_sec, 2))
            temperatures.append(round(estimated_temp, 2))

            # Optional: Draw rectangle around hot area
            frame_debug = frame.copy()
            offset_x = int(w * bar_width_ratio)
            # cv2.circle(frame_debug, (max_x + offset_x, max_y), 5, (0, 255, 0), 2)
            cv2.putText(frame_debug, f"{estimated_temp:.1f}C", (max_x + offset_x + 10, max_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.imshow("Hot Spot Detection", frame_debug)
            # time.sleep(0.5)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        frame_idx += 1

        if save_frames and (int(time_sec) % 5 == 0):
            frame_folder = os.path.join("output", "frames", os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(frame_folder, exist_ok=True)
            frame_filename = os.path.join(frame_folder, f"frame_{int(time_sec)}s.jpg")
            cv2.imwrite(frame_filename, frame_debug)

    cap.release()
    cv2.destroyAllWindows()
    return timestamps, temperatures

    # Save CSV
    csv_path = os.path.join("output", f"{name_no_ext}_temperature.xlsx")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Temperature (°C)"])
        writer.writerows(zip([round(x, 2) for x in timestamps], [round(x, 2) for x in temperatures]))
    print(f"[CSV] Saved to {csv_path}")

    # Save or show plot
    if save_plot or show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, temperatures, marker='o', color='red')
        plt.title(f'Temperature vs Time - {base_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        plt.tight_layout()
        if save_plot:
            plot_path = os.path.join("output", f"{name_no_ext}_temperature_plot.png")
            plt.savefig(plot_path)
            print(f"[PLOT] Saved to {plot_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Thermal video analysis")
    parser.add_argument('--folder', type=str, default='videos', help='Folder containing videos')
    parser.add_argument('--save-plot', action='store_true', help='Save plot image to output folder')
    parser.add_argument('--save-frames', action='store_true', help='Save frames every 5sec')
    parser.add_argument('--show-plot', action='store_true', help='Do not show plots interactively')
    args = parser.parse_args()

    folder = args.folder
    os.makedirs("output", exist_ok=True)
    videos = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]

    if not videos:
        print("No .mp4 videos found in folder:", folder)
        return

    all_data = {}

    for video in videos:
        video_path = os.path.join(folder, video)
        print(f"[INFO] Processing {video_path}")
        times, temps = process_video(video_path, save_plot=args.save_plot, save_frames=args.save_frames, show_plot=args.show_plot)
        base_name = os.path.splitext(os.path.basename(video))[0]
        df = pd.DataFrame({'Time (s)': times, 'Temperature (°C)': temps})
        all_data[base_name] = df

    # Save all to Excel
    excel_path = os.path.join("output", "temperature_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, df in all_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[EXCEL] Summary saved to {excel_path}")

if __name__ == "__main__":
    main()