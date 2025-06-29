import json
import os



class Tool:
    @staticmethod
    def readJSON(json_path):
        if not os.path.exists(json_path):
            msg = "The given json path does not exist."
            raise FileNotFoundError(msg)
        
        with open(json_path, 'r') as file:
            return json.load(file)
        
    
    @staticmethod
    def readBunchJSON(dir_path):

        # Validate directory path
        if not os.path.exists(dir_path):
            msg = "The given directory path does not exist."
            raise FileNotFoundError(msg)
        
        countries_polygons = {}

        countries = os.listdir(dir_path)
        # print(f"Available countries are {countries}.") # Debug

        for country in countries:
            json_path = f"{dir_path}/{country}"
            # use [:-5] to cancel *.json
            countries_polygons[country[:-5]] = Tool.readJSON(json_path)

        return countries_polygons
        


if __name__ == '__main__':
    dir_path = 'python\model\dataset\polygons'
    data = Tool.readBunchJSON(dir_path)
    print(data.keys())