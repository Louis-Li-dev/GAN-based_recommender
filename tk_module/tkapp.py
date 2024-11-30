
import tkinter as tk
import warnings
warnings.simplefilter(action='ignore')
from tkinter import messagebox
from utility.utility import  model_predicting
from PIL import Image, ImageTk

class SpotApp(tk.Tk):
    """
    A Tkinter application for predicting and displaying recommended travel spots based on user input.

    Attributes:
    ----------
    model_set : tuple
        A tuple containing the encoder and decoder PyTorch models.
    name_to_net_dict : dict
        A mapping of spot names to their corresponding network indices.
    images : dict
        A dictionary containing preloaded images of spots.

    Methods:
    -------
    set_mode():
        Updates the application mode based on user selection.

    add_spot():
        Adds spots to the listbox based on user input, validating against available data.

    clear_spots():
        Clears the input listbox and resets the application state.

    on_predict():
        Triggers the prediction process using the selected spots.

    display_results(img, recommended_spots):
        Displays the prediction results, including an image and recommended spots.

    display_spot_images(recommended_spots):
        Displays images of the recommended spots in a scrollable frame.

    on_frame_configure(event):
        Adjusts the scrollable region of the canvas when the inner frame is resized.
    """

    def __init__(self, model_set, name_to_net_dict, images):
        """
        Initializes the SpotApp application.

        Parameters:
        ----------
        model_set : tuple
            Encoder and decoder models for prediction.
        name_to_net_dict : dict
            A mapping of spot names to network indices.
        images : dict
            Preloaded images of spots.
        """
        super().__init__()

        self.model_set = model_set
        self.name_to_net_dict = name_to_net_dict
        self.images = images

        self.title("Spot List Input")
        self.geometry("1200x800")  # Increased size to fit the subplots
        self.minsize(1400, 800)    # Minimum size of the window

        self.mode = 1  # Default mode

        # Left Frame for Input Controls
        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label = tk.Label(self.left_frame, text="Enter spots (comma-separated):")
        self.label.pack(pady=10)

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self.left_frame, textvariable=self.entry_var)
        self.entry.pack(pady=10)

        self.submit_button = tk.Button(self.left_frame, text="Add Spot", command=self.add_spot)
        self.submit_button.pack(pady=10)

        self.spot_listbox = tk.Listbox(self.left_frame)
        self.spot_listbox.pack(pady=10)

        self.clear_button = tk.Button(self.left_frame, text="Clear Spots", command=self.clear_spots)
        self.clear_button.pack(pady=10)

        self.predict_button = tk.Button(self.left_frame, text="Predict", command=self.on_predict)
        self.predict_button.pack(pady=10)

        self.result_textbox = tk.Text(self.left_frame, height=10, wrap=tk.WORD)
        self.result_textbox.pack(pady=10, fill=tk.BOTH, expand=True)
   
        # Mode Selection
        self.mode_var = tk.IntVar(value=1)
        self.mode_frame = tk.Frame(self.left_frame)
        self.mode_frame.pack(pady=10)
        self.mode_label = tk.Label(self.mode_frame, text="Select Mode:")
        self.mode_label.pack(side=tk.LEFT, padx=5)
        self.mode1_radio = tk.Radiobutton(self.mode_frame, text="Mode 1", variable=self.mode_var, value=1,
                                          command=self.set_mode)
        self.mode1_radio.pack(side=tk.LEFT, padx=5)
        self.mode2_radio = tk.Radiobutton(self.mode_frame, text="Mode 2", variable=self.mode_var, value=2,
                                          command=self.set_mode)
        self.mode2_radio.pack(side=tk.LEFT, padx=5)
        self.mode2_radio.config(state=tk.DISABLED)

        # Right Frame for Displaying Image
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.right_frame)
        self.canvas.pack(pady=10, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.right_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar.place(relx=1, rely=0, relheight=1, anchor=tk.NE)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)

    def set_mode(self):
        self.mode = self.mode_var.get()

    def add_spot(self):
        spots = self.entry_var.get()
        spot_list = [spot.strip() for spot in spots.split(",")]

        invalid_spots = []
        valid_spots = []
        for spot in spot_list:
            if spot and spot in self.name_to_net_dict:
                valid_spots.append(spot)
                if spot in self.images and self.mode2_radio['state'] == tk.DISABLED:
                    self.mode2_radio.config(state=tk.NORMAL)
            else:
                invalid_spots.append(spot)
        
        if invalid_spots:
            messagebox.showerror("Input Error", f"Invalid spots: {', '.join(invalid_spots)}")
        
        for spot in valid_spots:
            self.spot_listbox.insert(tk.END, spot)
        
        self.entry_var.set("")

    def clear_spots(self):
        self.spot_listbox.delete(0, tk.END)
        self.result_textbox.delete(1.0, tk.END)
        self.mode2_radio.config(state=tk.DISABLED)
        self.mode_var.set(1)
        self.set_mode()

    def on_predict(self):
        spot_text_list = list(self.spot_listbox.get(0, tk.END))
        if not spot_text_list:
            messagebox.showerror("Input Error", "Please add some spots.")
            return

        try:
            img, recommended_spots = model_predicting(spot_text_list, self.model_set, self.name_to_net_dict)
            self.display_results(img, recommended_spots)
            if self.mode == 2 and any(spot in self.images for spot in recommended_spots):
                self.display_spot_images(recommended_spots)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, img, recommended_spots):
        self.canvas.delete("all")
        self.result_textbox.delete(1.0, tk.END)
        self.result_textbox.insert(tk.END, f"Recommended Spots:\n{', '.join(recommended_spots)}")

        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(240, 240, image=self.img_tk, anchor=tk.CENTER)


    def display_spot_images(self, recommended_spots):
        self.canvas.delete("all")
        self.scrollable_frame.destroy()
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)

        # Calculate dimensions for subplot grid based on number of spots
        num_spots = len(recommended_spots)
        rows = 2 * num_spots  # Each spot can have up to 4 images, so 2 rows per spot
        cols = 2  # 2 columns for each spot's images

        # Define the target size for images
        target_size = (256, 256)  # Set desired width and height

        for i, spot in enumerate(recommended_spots):
            if spot in self.images:
                image = self.images[spot]

                for idx, (style_name, img) in enumerate(image.items()):
                    row = 2 * i + idx // 2
                    col = idx % 2

                    # Resize the image to the target size
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # High-quality resizing
                    
                    # Convert to ImageTk for displaying in Tkinter
                    img_tk = ImageTk.PhotoImage(resized_img)
                    label = tk.Label(self.scrollable_frame, image=img_tk)
                    label.image = img_tk  # Keep a reference to prevent garbage collection
                    label.grid(row=row, column=col, padx=10, pady=10)

        self.scrollable_frame.update_idletasks()  # Update frame to ensure proper layout
        self.canvas.update_idletasks()  # Update canvas to ensure scroll region is correct
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))