import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import(accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            classification_report)
# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_excel("C:\\Users\\ANANGSHA\\Documents\\cybersecurity_intrusion_detection_dataset.csv.xlsx")
print(df.head())
print(df.tail())
print(df.dtypes)

#-------------------------------------------------
#Features and Target 
#-------------------------------------------------
x = df[['Packet_Size', 'Request_Frequency']]
y = df["Attack_Type"]
print(df['Attack_Type'].unique())

#-------------------------------------------------
#Splitting Test and Train datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=50, stratify=y)

#-------------------------------------------------
#Feature Scaling
#-------------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#-------------------------------------------------
#SVM model creating, fitting and prediction
#-------------------------------------------------
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(x_train_scaled, y_train)
linear_pred = linear_svm.predict(x_test_scaled)

#-------------------------------------------------
#Metrics Evaluation for Linear SVM
#-------------------------------------------------
accuracy_linear = accuracy_score(y_test, linear_pred)
precision_linear = precision_score(y_test, linear_pred, average='weighted')
recall_linear = recall_score(y_test, linear_pred, average='weighted')
f1_linear = f1_score(y_test, linear_pred, average='weighted')

print("Accuracy:",accuracy_linear)
print("Precision:",precision_linear)
print("Recall:",recall_linear)
print("F1:",f1_linear)

# -------------------------------------------------
# Save Models
# -------------------------------------------------
joblib.dump(linear_svm, "linear_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
