# pipeline/transformer.py
import pandas as pd
class Transformer:

    def __init__(self):
        pass


    def remove_date_time(self, data: pd.DataFrame) -> pd.DataFrame:

        data['Date'] = data['Date'].dt.date
        
        return data
