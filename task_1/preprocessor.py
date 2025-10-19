import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_titanic(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    data = df.copy()
    
    # Feature Engineering
    # Extract title from name
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Create IsAlone feature
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Handle missing values
    
    # Age: Fill with median age by Title and Pclass
    for title in data['Title'].unique():
        for pclass in data['Pclass'].unique():
            mask = (data['Title'] == title) & (data['Pclass'] == pclass)
            median_age = data[mask]['Age'].median()
            if pd.notna(median_age):
                data.loc[mask & data['Age'].isna(), 'Age'] = median_age
    
    data['Age'].fillna(data['Age'].median(), inplace=True)
    
    # Embarked: Fill with mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Fare: Fill with median
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Cabin: Create binary feature for cabin known/unknown
    data['HasCabin'] = data['Cabin'].notna().astype(int)
    
    # Create age groups
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
    
    # Create fare bins
    data['FareBin'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], 
                               duplicates='drop')
    
    # Encode categorical variables
    le = LabelEncoder()
    
    categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareBin']
    for col in categorical_cols:
        data[col + '_Encoded'] = le.fit_transform(data[col].astype(str))
    
    # Select features for final dataset
    features_to_keep = [
        'PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 
        'Fare', 'FamilySize', 'IsAlone', 'HasCabin',
        'Sex_Encoded', 'Embarked_Encoded', 'Title_Encoded', 
        'AgeGroup_Encoded', 'FareBin_Encoded'
    ]
    
    # Filter columns that exist in the dataset
    features_to_keep = [col for col in features_to_keep if col in data.columns]
    
    # Create final preprocessed dataset
    preprocessed_data = data[features_to_keep]
    
    # Save to CSV
    preprocessed_data.to_csv(output_file, index=False)
    
    print(f"Preprocessing complete")
    print(f"Original shape: {df.shape}")
    print(f"Preprocessed shape: {preprocessed_data.shape}")
    print(f"\nMissing values in preprocessed data:")
    print(preprocessed_data.isnull().sum())
    print(f"\nPreprocessed data saved to: {output_file}")
    
    return preprocessed_data

if __name__ == "__main__":
    input_path = "../data/titanic/train.csv"
    output_path = "../data/titanic_preprocessed.csv"
    
    preprocessed_df = preprocess_titanic(input_path, output_path)
    
    # Display first few rows
    print("\nFirst 5 rows of preprocessed data:")
    print(preprocessed_df.head())