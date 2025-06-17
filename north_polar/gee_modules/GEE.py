import ee



class GEE:
    def __init__(self, project_id:str):
        self.project_id = project_id
    
    
    def connect(self):
        """
        Connect to Google Earth Engine
        """
        try:
            ee.Authenticate()
            ee.Initialize(project=self.project_id)

        except:
            pass