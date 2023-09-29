import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp


class App:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x521")
        self.root.configure(bg="#5D0016")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 50)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.left = False
        self.right = False

        self.segmentor = SelfiSegmentation()
        self.fpsReader = cvzone.FPS()

        self.imgList = []
        self.indexImg = 0

        self.frame = tk.Frame(self.root)
        self.frame.pack(anchor=tk.CENTER)  # Align the frame at the center

        self.canvas = tk.Canvas(self.frame, width=1280, height=521)
        self.canvas.pack()

        self.change_bg_button1 = tk.Button(
            self.frame,
            text="PREVIOUS",
            command=self.change_background_prev,
            bg="#5D0016",
            fg="gray",
            padx=10,
            pady=10,
            width=10,
        )
        self.change_bg_button1.place(
            x=880, y=480
        )  # Position the button above the input window

        self.change_bg_button2 = tk.Button(
            self.frame,
            text="NEXT",
            command=self.change_background_next,
            bg="#5D0016",
            fg="gray",
            padx=10,
            pady=10,
            width=10,
        )
        self.change_bg_button2.place(
            x=990, y=480
        )  # Position the button above the input window

        self.add_bg_button = tk.Button(
            self.frame,
            text="SET BACKGROUND",
            command=self.add_background,
            bg="#5D0016",
            fg="gray",
            padx=10,
            pady=10,
            width=15,
        )
        self.add_bg_button.place(
            x=240, y=480
        )  # Position the button above the input window

        self.virtual_bg_label = tk.Label(
            self.root,
            text="VIRTUAL BACKGROUND",
            bg="#5D0016",
            fg="gray",
            font=("Times New Roman", 38),
        )

        self.virtual_bg_label.pack(side=tk.TOP, pady=10)

        self.virtual_bg_text = tk.Label(
            self.root,
            text="Click the 'SET BACKGROUND' button to choose custom background images. Select one or multiple image files (JPEG or PNG) from your device.\nUse the 'PREVIOUS' and 'NEXT' buttons to cycle through the selected background images. This allows you to choose the desired virtual background for your video.\nThe application uses your device's camera to capture real-time video. The video feed will be displayed on the screen within the application window.\nThe application can recognize hand gestures. Move your hand to the left to switch to the previous background image and move it to right to switch to the next background image.",
            bg="#5D0016",
            fg="gray",
            font=("Times New Roman", 13),
            justify=tk.LEFT,
        )

        self.virtual_bg_text.pack(side=tk.TOP, padx=55, pady=5)

        self.update_frame()

    def update_frame(self):
        _, img = self.cap.read()

        if self.imgList and self.indexImg < len(
            self.imgList
        ):  # Check if imgList is not empty and index is within range
            imgOut = self.segmentor.removeBG(
                img, self.imgList[self.indexImg], threshold=0.4
            )
        else:
            imgOut = img

        imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
        _, imgStacked = self.fpsReader.update(imgStacked)
        print(self.indexImg)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        multiLandmarks = results.multi_hand_landmarks

        if multiLandmarks:
            handPoints = []
            for handLms in multiLandmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                for idx, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handPoints.append((cx, cy))

            if not self.left and handPoints[8][0] < 220:
                self.left = True
                if self.indexImg > 0:
                    self.indexImg -= 1
                    print("Left")
            if not self.right and handPoints[8][0] > 420:
                self.right = True
                if self.indexImg < len(self.imgList) - 1:
                    self.indexImg += 1
                    print("Right")

            if handPoints[8][0] > 220 and handPoints[8][0] < 420:
                self.left = False
                self.right = False

        self.photo = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(imgStacked, cv2.COLOR_BGR2RGB))
        )
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(1, self.update_frame)

    def change_background_prev(self):
        self.indexImg -= 1
        if self.indexImg < 0:
            self.indexImg = len(self.imgList) - 1

    def change_background_next(self):
        self.indexImg += 1
        if self.indexImg >= len(self.imgList):
            self.indexImg = 0

    def add_background(self):
        filetypes = (("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        selected_files = filedialog.askopenfiles(filetypes=filetypes)

        if selected_files:
            self.imgList = []  # Clear the existing background images
            for file in selected_files:
                img_path = file.name
                img = cv2.imread(img_path)
                rsimgBg = cv2.resize(img, (640, 480))
                self.imgList.append(rsimgBg)
            self.indexImg = 0  # Reset the index

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.start()
