from datasets import load_dataset

def getData():
    ds = load_dataset("Gepe55o/mountain-ner-dataset", split=['train[:10%]','test[:10%]'])
    return ds
