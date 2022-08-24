import pandas as pd
import numpy as np


class BasicCleaner:
    def preprocessing(self, df):
        df = df.copy(deep=True)
        df["first_name"], df["last_name"] = self.extract_first_last_name(df.name)
        df["title"] = self.extract_title(df.name)
        df["sex"] = self.binarize_sex(df.sex)
        return df.drop("name", axis=1)
        
    @staticmethod
    def extract_first_last_name(names):
        first_name = [name.split('.')[0].split(',')[0].strip() for name in names]
        last_names = [name.split('.')[1].strip() for name in names]
        return (first_name, last_names)
    
    @staticmethod
    def extract_title(names):
        uncommon_titles = 'Rev,Dr,Col,Major,Mlle,Ms,Sir,Capt,Mme,Jonkheer,Lady,the Countess,Don,Dona'.lower().split(',')
        titles = [i.rsplit(', ')[1].rsplit('.')[0].lower() for i in names]  
        titles = ['uncommon' if i in uncommon_titles else i for i in titles]  # Group uncommon titles
        return titles

    @staticmethod
    def binarize_sex(gender):
        return [1 if i == 'male' else 0 for i in gender]
    
    @staticmethod
    def expand_age(df):
        df = df.copy(deep=True)
        df['infant'] = [1 if i < 1 else 0 for i in df.age] 
        df['child'] = [1 if i in range(1, 13) else 0 for i in df.age]
        df['teen'] = [1 if i in range(13, 19) else 0 for i in df.age]
        df['young_adult'] = [1 if i in range(19, 30) else 0 for i in df.age]
        df['adult'] = [1 if i in range(30, 40) else 0 for i in df.age]
        df['age40+'] = [1 if i >= 40 else 0 for i in df.age]
        return df.drop('age', axis=1)