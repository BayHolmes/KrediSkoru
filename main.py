from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

app = Flask(__name__)
# Flask uygulamasını başlatır. Bu, web sunucusunu başlatmanın bir yoludur.

app.secret_key = '126354'
# Bu anahtar, Flask'ın oturum verilerini güvenli bir şekilde saklamasını sağlar.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['existing_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
                'savings', 'employment_since', 'installment_rate', 'personal_status_sex',
                'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
                'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                'credit_risk']
df = pd.read_csv(url, names=column_names, delimiter=' ', header=None)
# Veriyi CSV dosyasından okur ve bir pandas DataFrame'ine dönüştürür.

# Kategorik değişkenleri sayısal değerlere dönüştür
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
# Kategorik özellikleri sayısal değerlere çevirir.

X = df.drop(columns=['credit_risk'])
y = df['credit_risk']
# Hedef değişkeni ve özellikleri ayırır.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Veriyi eğitim ve test setlerine ayırır.

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# Random Forest modelini eğitir.

@app.route('/')
def home():
    existing_checking_values = ['A11', 'A12', 'A13', 'A14']
    return render_template('home.html', existing_checking_values=existing_checking_values)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Kullanıcının girdiğini formdan alır.

    df_input = pd.DataFrame(columns=X.columns)
    for column in df_input.columns:
        if column in data:
            df_input.loc[0, column] = data[column]
        else:
            df_input.loc[0, column] = 0
    # Kullanıcıdan alınan veriyi bir DataFrame'e çevirir.

    for column, le in label_encoders.items():
        if column in df_input.columns:
            try:
                df_input[column] = le.transform(df_input[column]).tolist()
            except ValueError:
                df_input[column] = le.transform([le.classes_[0]] * len(df_input[column])).tolist()
    # Kategorik değişkenleri sayısal değerlere dönüştürür.

    # Modeli kullanarak bir tahmin yap
    prediction = clf.predict(df_input)

    # Her bir özelliğin skoru üzerindeki etkisini hesapla
    scores = {}
    for column in df_input.columns:
        df_temp = df_input.copy()
        df_temp[column] = X[column].mean()  # Değişikliği sadece bu sütuna uygula
        prediction_without_feature = clf.predict(df_temp)
        scores[column] = int(
            abs(prediction[0] - prediction_without_feature[0]))  # convert ndarray to list and int64 to int

    # Skorları session'da sakla ve feature_importances sayfasında kullan
    session['scores'] = scores

    if prediction[0] == 1:
        prediction_message = 'Tahmini kredi risk skoru: 1. <br><br> Bu, musterinin kredi geri odeme performansinin iyi oldugunu gostermektedir.'
    elif prediction[0] == 2:
        prediction_message = 'Tahmini kredi risk skoru: 2. <br><br> Bu, musterinin kredi geri odeme performansinin kotu oldugunu gostermektedir.'
    prediction_message += '<br><br> <a href="/feature_importances">Ozelliklerin Onem Derecesini Goruntule</a>'
    # Tahmini ve anlamını oluşturur.

    return prediction_message, 200, {'Content-Type': 'text/html'}
    # Tahmini ve anlamını HTML'de döndürür.


@app.route('/feature_importances')
def feature_importances():
    feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_imp = feature_imp.to_frame(name='Feature Importance')



    return feature_imp.to_html(), 200, {'Content-Type': 'text/html'}
    # Özelliklerin önemini ve kullanıcıya sağladığı skoru HTML'de döndürür.


if __name__ == '__main__':
    app.run(debug=True)
    # Uygulamayı çalıştırır ve debug modunda başlatır.
