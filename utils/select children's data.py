import pandas as pd 
data = pd.read_csv("Data_Entry_2017.csv")
children_data = data[data["Patient Age"]<16]
children_data.to_csv("children_data.csv")