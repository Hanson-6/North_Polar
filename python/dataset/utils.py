import json
import os



class Tool:
    @staticmethod
    def readJSON(json_path):
        """
        给定一个json文件路径，读取并返回json内容。

        参数：
            - json_path (str): json文件的路径

        返回：
            - dict: 解析后的json内容
        """
        with open(json_path, 'r') as file:
            return json.load(file)
        
    
    @staticmethod
    def readBunchJSON(dir_path):
        """
        给定一个目录路径，读取该目录下所有的json文件，并将其内容存储在一个字典中。

        参数：
            - dir_path (str): 包含json文件的目录路径

        返回：
            - dict: 包含所有json文件内容的字典，键为文件名（不含扩展名），值为解析后的json内容
        """
        
        countries_polygons = {}

        countries = os.listdir(dir_path)
        # print(f"Available countries are {countries}.") # Debug

        for country in countries:
            json_path = f"{dir_path}/{country}"
            # use [:-5] to cancel *.json
            countries_polygons[country[:-5]] = Tool.readJSON(json_path)

        return countries_polygons