import pandas as pd
from utils import decontracted, other_processing

def read(projection_path, reports_path, image_path):
    # Path to the CSV file (modify if needed)
    projections = projection_path
    # Read the CSV file into a DataFrame
    df = pd.read_csv(projections)
    frontal_df = df[df["projection"] == "Frontal"]  # Assuming "projection" is the column name

    # Read in the reports data
    reports = pd.read_csv(reports_path)
        
    # Merge the projections and reports data on the UID column
    reports = pd.merge(frontal_df, reports, on='uid')

    selected_reports = reports[['uid', 'filename', 'findings','impression']]
    selected_reports['filename'] = image_path + selected_reports['filename']

    return selected_reports

def clean(selected_reports):
    # Impute NaN values in 'findings' using 'impression' (using fillna)
    frontal_reports = selected_reports.fillna({'findings': selected_reports['impression']})
    data = frontal_reports[['uid', 'filename', 'findings']]
    data = data.dropna(subset=['findings'])
    data['findings'] = data['findings'].str.lower()
    data['findings']= data['findings'].apply(decontracted)
    data['findings']= data['findings'].apply(other_processing)

    data.to_csv("/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/cleaned_df.csv", index = False)

if __name__=="__main__":
    projection_path = "/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/indiana_projections.csv"
    reports_path = '/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/indiana_reports.csv'
    image_path = "/home/dasith/Documents/Personal/Academics/chestXtray/Datasets/indiana/images/images_normalized/"
    reports = read(projection_path, reports_path, image_path)
    clean(reports)