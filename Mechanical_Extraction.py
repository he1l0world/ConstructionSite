import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

class MechanicalLegendExtractor:
    def __init__(self, image_path: str):
        """Initialize the extractor with an image path."""
        self.binary_image = None
        self.image_array = None
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}")

        self.intensity_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def preprocess_image(self) -> np.ndarray:
        """Preprocess the image to improve text and symbol detection."""
        # Increase contrast
        threshold_value = 128
        self.image_array = np.array(self.intensity_image)
        self.binary_image = (self.image_array > threshold_value).astype(np.uint8) * 255

        return self.binary_image

    def calculate_horizontal_intensity(self):
        self.horizontal_projection = np.sum(self.image_array, axis=1)
        self.horizontal_line = np.argmax(self.image_array, axis=1)

        # Plot the horizontal projection
        plt.figure(figsize=(10, 6))
        plt.plot(self.horizontal_projection, label='Horizontal Projection')
        plt.xlabel('Row Index')
        plt.ylabel('Sum of Pixel Intensities')
        plt.title('Horizontal Projection of Image')
        plt.legend()
        plt.show()

    def calculate_vertical_intensity(self):
        # Compute vertical projection (sum of pixel intensities along columns)
        self.vertical_projection = np.sum(self.image_array, axis=0)

        # Plot the vertical projection
        plt.figure(figsize=(10, 6))
        plt.plot(self.vertical_projection, label='Vertical Projection')
        plt.xlabel('Column Index')
        plt.ylabel('Sum of Pixel Intensities')
        plt.title('Vertical Projection of Image')
        plt.legend()
        plt.show()

    def white_pixel_based_row_segmentation(self, threshold=0, distance = 20):
        """
        Segments rows by identifying lines where all pixels are white (255).
        """
        row_sums = np.sum(self.binary_image < 255, axis=1)  # Count non-white pixels in each row
        row_indices = np.where(row_sums <= threshold)[0]  # Identify rows with all white pixels
        segments = []

        start = row_indices[0]
        for i in range(1, len(row_indices)):
            if row_indices[i] != row_indices[i - 1] + 1 and row_indices[i] - start > distance:
                segments.append((start, row_indices[i]))
                start = row_indices[i]
        # segments.append((start, row_indices[-1]))

        return segments

    def col_segmentation(self):
        # Find where to start looking for the vertical
        mean_value_index = np.argmin(self.vertical_projection >= np.mean(self.vertical_projection))

        # Find the maximum intensity index after the mean value index
        max_intensity_index = np.argmax(self.vertical_projection[mean_value_index:]) + mean_value_index


        # Split the image into two columns at the point of minimum intensity
        self.column_split = max_intensity_index

    def draw_labeled_segmentation(self):
        # Draw row labels
        plt.imshow(self.binary_image, cmap='gray', aspect='auto')
        for row in self.row_segments:
            plt.hlines(y=row, xmin=0, xmax=self.binary_image.shape[1], colors='red', linestyles='dashed')

        # Draw column split line
        plt.vlines(x=self.column_split, ymin=0, ymax=self.binary_image.shape[0], colors='blue', linestyles='dashed')

        plt.title("Image with Labeled Rows and Two-Column Segmentation")
        plt.savefig("segmentation_label.png")
        plt.show()

    def process_table(self) :
        """Process the entire table with improved cell detection."""
        # Preprocess image
        self.preprocess_image()
        self.calculate_horizontal_intensity()
        self.calculate_vertical_intensity()

        # Detect edges
        self.row_segments = self.white_pixel_based_row_segmentation(self.horizontal_line, 20)
        self.col_segmentation()

        #draw the labeled segmentations
        self.draw_labeled_segmentation()



    def save_json(self, output_path: str) :
        """Process the table and save results to path."""
        # Ensure output directories exist
        from pathlib import Path
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        symbols = []
        descriptions = []
        for i, (start_row, end_row) in enumerate(self.row_segments):
            # Extract the row
            row_image = self.binary_image[start_row:end_row, :]

            # Split the row into two columns using the previously identified column split
            first_column = row_image[:, :self.column_split]
            second_column = row_image[:, self.column_split:]

            # Save the symbol (first column) as an image
            symbol_path = output_dir / f"symbol_{i + 1}.png"
            cv2.imwrite(str(symbol_path), first_column)
            symbols.append(symbol_path)

            # Recognize text in the second column using Tesseract OCR
            recognized_text = pytesseract.image_to_string(second_column, config="--psm 6")
            descriptions.append(recognized_text.strip())

        # Save results into a structured format
        import pandas as pd

        self.results = pd.DataFrame({
            "Symbol Image Path": symbols,
            "Description": descriptions
        })

        self.results.to_json('result.json', default_handler=str, indent=True)

        print(self.results)

    def display(self):
        # Convert DataFrame to a Matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 2))  # Adjust figure size as needed
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=self.results.values, colLabels=self.results.columns, cellLoc='center', loc='center')
        plt.savefig("results_image.png")
        plt.show()


def extract_mechanical_legend(image_path: str, output_path: str) -> None:
    try:
        extractor = MechanicalLegendExtractor(image_path)
        extractor.process_table()
        extractor.save_json(output_path)
        extractor.display()
    except Exception as e:
        print(f"Error processing image: {e}")


# Example usage with debugging
if __name__ == "__main__":
    output_path = "extracted_symbols"
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    image_path = 'Mechanical Legend.png'
    extract_mechanical_legend(image_path,output_path)