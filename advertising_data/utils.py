import pickle
import numpy as np
import config
import json

class  Sales():
    def __init__(self):
        pass

    def __load_data(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model = pickle.load(f)

        self.col_names = self.model.feature_names_in_

    def get_predicted_sales(self,tv, radio, newspaper):
        self.__load_data()
        print('tv,radio,newspaper',tv,radio,newspaper)

        test_array = np.zeros((1,self.model.n_features_in_))
        test_array[0,0] = tv
        test_array[0,1] = radio
        test_array[0,2] = newspaper

        predicted_sales = self.model.predict(test_array)
        print("predicted_sales :",predicted_sales)

        return predicted_sales