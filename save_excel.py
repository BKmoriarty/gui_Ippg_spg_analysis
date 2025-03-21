import pandas as pd


class DataExporter:
    def __init__(self):
        # Initialize an empty list to store data
        self.data = []

    def add_record(self, record):
        """Add a single record (dictionary) to the data list."""
        if isinstance(record, dict):
            self.data.append(record)
        else:
            raise ValueError("Record must be a dictionary")

    def add_multiple_records(self, records):
        """Add multiple records at once."""
        if not all(isinstance(r, dict) for r in records):
            raise ValueError("All records must be dictionaries")
        self.data.extend(records)

    def save_to_excel(self, filename="output.xlsx"):
        """Save the collected data to an Excel file."""
        if not self.data:
            print("No data to save!")
            return

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(self.data)

        # Save to Excel
        try:
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")

    def clear_data(self):
        """Clear the stored data."""
        self.data = []
        print("Data cleared")


# Example usage
if __name__ == "__main__":
    # Create an instance of DataExporter
    exporter = DataExporter()

    # Add some sample data
    exporter.add_record({"Name": "Alice", "Age": 25, "City": "New York"})
    exporter.add_record({"Name": "Bob", "Age": 30, "City": "London"})

    # Add multiple records
    more_data = [
        {"Name": "Charlie", "Age": 35, "City": "Tokyo"},
        {"Name": "Diana", "Age": 28, "City": "Paris"}
    ]
    exporter.add_multiple_records(more_data)

    # Save to Excel
    exporter.save_to_excel("my_data.xlsx")

    # Clear data if needed
    # exporter.clear_data()
