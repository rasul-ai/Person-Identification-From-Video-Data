import os
import argparse
import cv2

def main():
    # Parse all args
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to source video')
    parser.add_argument('dest_folder', type=str, help='Path to destination folder')
    args = parser.parse_args()

    # Get file path for desired video and where to save frames locally
    cap = cv2.VideoCapture(args.source)
    path_to_save = os.path.abspath(args.dest_folder)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if not cap.isOpened():
        print('Error: Unable to open video file.')
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Video duration in seconds

    print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f} seconds")

    # Interval to capture frames (in seconds)
    frame_interval = 1 / 2  # Extract 2 frames per second

    # Frame index to capture
    frame_indices = [int(i * fps * frame_interval) for i in range(int(duration * 2))]

    current_frame = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if current_frame in frame_indices:
            # Save frame as a jpg file
            name = f'frame{saved_frames + 1:03d}.jpg'
            print(f'Creating: {name}')
            cv2.imwrite(os.path.join(path_to_save, name), frame)
            saved_frames += 1

        current_frame += 1

    # Release capture
    cap.release()
    print(f'Done! Extracted {saved_frames} frames.')

if __name__ == '__main__':
    main()
