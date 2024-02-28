import pickle
import numpy as np
import config
import json

class  MedicalInsurance():
    def __init__(self):
        pass

    def __load_data(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model = pickle.load(f)

        with open(config.COLUMN_DATA_JSON, 'r') as f:
            self.col_data = json.load(f)

        self.col_names = self.model.feature_names_in_


    def get_predicted_charges(self,age,gender,bmi,children,smoker,region):

        self.__load_data()
        print('age,gender,bmi,children,smoker,region',age,gender,bmi,children,smoker,region)

        region_index = np.where(self.col_names == 'region_'+ region)[0][0]
        # print("region_index",region_index)

        gender = self.col_data['gender'][gender]
        smoker = self.col_data['smoker'][smoker]

        test_array = np.zeros((1,self.model.n_features_in_))
        test_array[0,0] = age
        test_array[0,1] = gender
        test_array[0,2] = bmi
        test_array[0,3] = children
        test_array[0,4] = smoker
        test_array[0,region_index] = 1

        predicted_price = self.model.predict(test_array)
        print("predicted_price :",predicted_price)

        return predicted_price
