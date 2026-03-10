if __name__ == "__main__":
    from data_loader import DataLoader

    # Example usage
    data_loader = DataLoader(data_set="a")
    df = data_loader.process_dataset()
    print(df.head())