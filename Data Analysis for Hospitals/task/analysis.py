import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
general_path = "C:/Users/ravit/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/test/general.csv"
prenatal_path = "C:/Users/ravit/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/test/prenatal.csv"
sports_path = "C:/Users/ravit/PycharmProjects/Data Analysis for Hospitals/Data Analysis for Hospitals/test/sports.csv"

# Load datasets
general = pd.read_csv(general_path)
prenatal = pd.read_csv(prenatal_path)
sports = pd.read_csv(sports_path)

# Rename columns in prenatal and sports to match general
prenatal.columns = general.columns
sports.columns = general.columns

# Merge datasets
merged_df = pd.concat([general, prenatal, sports], ignore_index=True)

# Drop 'Unnamed: 0' if it exists
if 'Unnamed: 0' in merged_df.columns:
    merged_df = merged_df.drop(columns=['Unnamed: 0'])

# Handle missing values and fix gender column
merged_df['gender'] = merged_df['gender'].replace({'male': 'm', 'female': 'f', 'man': 'm', 'woman': 'f'})
merged_df['gender'].fillna('f', inplace=True)

# Replace NaN values in specified columns with zeros
columns_to_fill = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']
merged_df[columns_to_fill] = merged_df[columns_to_fill].fillna(0)

# 1. Most common age range (Histogram)
age_bins = [0, 15, 35, 55, 70, 80]
age_labels = ['0-15', '15-35', '35-55', '55-70', '70-80']
merged_df['age_range'] = pd.cut(merged_df['age'], bins=age_bins, labels=age_labels, right=False)

# Plot histogram for age range
age_range_counts = merged_df['age_range'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
age_range_counts.plot(kind='bar', color='skyblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age Range')
plt.ylabel('Number of Patients')
plt.show()

# Find the most common age range
most_common_age_range = age_range_counts.idxmax()
print(f"The answer to the 1st question: {most_common_age_range}")

# 2. Most common diagnosis (Pie chart)
diagnosis_counts = merged_df['diagnosis'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(diagnosis_counts, labels=diagnosis_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Most Common Diagnosis')
plt.show()

# Find the most common diagnosis
most_common_diagnosis = diagnosis_counts.idxmax()
print(f"The answer to the 2nd question: {most_common_diagnosis}")

# 3. Violin plot for height distribution by hospitals
plt.figure(figsize=(10, 6))
sns.violinplot(x='hospital', y='height', data=merged_df, inner='quartile')
plt.title('Height Distribution by Hospitals')
plt.show()

# Answer to the third question (explanation about the gap and peaks)
print("The answer to the 3rd question: The gap in height distribution likely arises due to the different specialties of the hospitals. "
      "Sports hospitals tend to have taller patients due to athletes, while general hospitals cater to a more diverse population, including children. "
      "Thus, the peaks correspond to relatively shorter patients (including children) and taller individuals (athletes).")
