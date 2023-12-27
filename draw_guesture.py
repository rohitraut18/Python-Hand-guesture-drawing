import numpy as np
import cv2
from collections import deque
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to convert normalized coordinates to pixel coordinates
def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    pixel_x = int(normalized_x * image_width)
    pixel_y = int(normalized_y * image_height)
    return pixel_x, pixel_y

# Main function
def main():
    # Initialize hand tracking module
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Create a deque to store finger tip points
    pts = deque(maxlen=64)

    # Variable to track drawing status
    drawing = False

    while cap.isOpened():
        # Read a frame from the webcam
        _, image = cap.read()

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the hand landmarks
        results_hand = hands.process(image_rgb)

        # Convert the image back to BGR format
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Check if hand landmarks are detected
        if results_hand.multi_hand_landmarks:
            # Extract landmarks of the first detected hand
            hand_landmarks = results_hand.multi_hand_landmarks[0]

            # Create a dictionary to store landmark coordinates
            idx_to_coordinates = {}

            # Loop through all landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                landmark_px = normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0]
                )
                # Store the coordinates in the dictionary
                idx_to_coordinates[idx] = landmark_px

            # Check if the index finger is pointed up
            if 8 in idx_to_coordinates and 6 in idx_to_coordinates:
                # Calculate distance between index and middle finger tips
                distance = calculate_distance(
                    idx_to_coordinates[8], idx_to_coordinates[6]
                )

                # Start drawing if the index finger is pointed up and fingers are apart
                if (
                    idx_to_coordinates[8][1] < idx_to_coordinates[6][1]
                    and distance > 30  # Adjust this threshold based on your hand size and orientation
                ):
                    drawing = True
                # Stop drawing if the fist is closed
                else:
                    drawing = False
                    # Clear the drawing
                    pts.clear()

            # Draw lines if drawing is enabled
            if drawing and 8 in idx_to_coordinates:
                pts.appendleft(idx_to_coordinates[8])  # Index Finger
                for i in range(1, len(pts)):
                    if pts[i - 1] is not None and pts[i] is not None:
                        # Draw lines between consecutive finger tip points
                        thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
                        cv2.line(image, pts[i - 1], pts[i], (0, 0, 255), thick)

        # Display the image
        cv2.imshow("Hand Tracking", image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release resources
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
