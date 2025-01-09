from datasets import load_dataset
import os

# Function to get data from hugslib and save to files for furter access
def getData():
    ds = load_dataset("Gepe55o/mountain-ner-dataset", split=['train[:10%]','test[:10%]'])
    os.makedirs("NER/data", exist_ok=True)
    export_to_file("./NER/data/train.txt", ds[0])
    export_to_file("./NER/data/test.txt", ds[1])
    return ds

# Function to save database to file
    # Opens file and writes data where:
    #   - Each line contains different database entry
    #   - Line consists of: [number of words]+[splitted sentence]+[Tokens]
def export_to_file(export_file_path, data):
    with open(export_file_path, "w", encoding="utf-8") as f:
        for record in data:
            ner_tags = record["tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )
getData()