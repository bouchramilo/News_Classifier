from datasets import load_dataset
import pandas as pd

def load_data():
    
    print("Loading dataset SetFit/ag_news")
    dataset = load_dataset("SetFit/ag_news")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    print("Dataset loaded successfully.")
    return train_df, test_df