def main():
    import torch
    import os
    import pickle
    from l2reddit.databalancer import DataBalancer
    from l2reddit.dataprocessor import DataProcessor
    import pandas 
 
    data_processor = DataProcessor('google/bigbird-roberta-base')
    data_processor.discover_chunks('/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/balanced/seed_42/non_europe_data')

    #Load the entire dataset as a "test_dataset" since we're going to measure performance with 5FCV
    train_dataset, test_dataset = data_processor.get_train_test_datasets(split_by_chunks=True, seed=42, sequence_length=2048, train_size=0.0)
    test_dataset.extract_encodings_and_labels_from_chunks()

    initial = test_dataset.get_dataframe()
    data = initial['text'].to_frame()
    labels = initial['label'].to_frame()
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'wb') as f:
        pickle.dump(data, f)
        f.close()
    
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'wb') as f:
        pickle.dump(labels, f)
        f.close()

if __name__ == "__main__":
    main()