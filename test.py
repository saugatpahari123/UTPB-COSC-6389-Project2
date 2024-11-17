import tkinter as tk
from AppKit import NSOpenPanel

def test_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls;*.xlsx")],
            title="Select a Dataset File"
        )
        if file_path:
            print(f"File selected: {file_path}")
        else:
            print("No file selected.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_file_dialog()
