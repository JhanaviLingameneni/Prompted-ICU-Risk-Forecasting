if __name__ == "__main__":
    from data_loader import DataLoader

    # Example usage
    data_loader = DataLoader(data_set="a")
    df, y = data_loader.process_dataset()
    print(df.head())

    with open("temp.txt", "w", encoding="utf-8") as temp_file:
        temp_file.write("df.head():\n")
        temp_file.write(df.head().to_string())
        temp_file.write("\n\ncolumns:\n")
        for column in df.columns:
            temp_file.write(f"{column}\n")