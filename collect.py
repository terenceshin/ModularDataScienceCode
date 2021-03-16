import pandas as pd

def gbq_to_df(query, project):
    """
    Writes query results from GBQ to a DataFrame
    
    Inputs:
        query (str): The code to query GBQ
        project (str): the project id that you want to query in
        
    Returns:
        DataFrame with results from query
    """
    df = pd.read_gbq(query, project_id = project, reauth=False, dialect='standard')
    return df