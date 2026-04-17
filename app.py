from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

app = Flask(__name__)
app.secret_key = "secret123"

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

ADMIN_USER = "admin"
ADMIN_PASS = "1234"

# ---------------- LOGIN ----------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USER and request.form['password'] == ADMIN_PASS:
            session['admin'] = True
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid Login")
    return render_template('login.html')


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'admin' not in session:
        return redirect(url_for('login'))

    df = pd.read_csv("dataset.csv")

    # Rename columns
    df.columns = ["timestamp","vibration","temperature","pressure","rms_vibration","mean_temp","fault"]

    fig1 = px.bar(df, x="timestamp", y="vibration", title="Vibration Over Time")
    fig2 = px.pie(df, names="fault", title="Fault Distribution")
    fig3 = px.scatter(df, x="temperature", y="pressure",
                      color="fault", size="vibration",
                      title="Temperature vs Pressure")
    fig4 = px.line(df, x="timestamp", y="temperature",
                   title="Temperature Trend")

    return render_template('dashboard.html',
                           graph1=fig1.to_html(full_html=False),
                           graph2=fig2.to_html(full_html=False),
                           graph3=fig3.to_html(full_html=False),
                           graph4=fig4.to_html(full_html=False))


# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            data = [
                float(request.form['vibration']),
                float(request.form['temperature']),
                float(request.form['pressure']),
                float(request.form['rms_vibration']),
                float(request.form['mean_temp'])
            ]

            prediction = model.predict([data])[0]

            if prediction == 0:
                result = "✅ Good Condition"
            elif prediction == 1:
                result = "❌ Fault Detected"
            else:
                result = "⚠ Moderate Condition"

            return render_template('predict.html', prediction_text=result)

        except Exception as e:
            return render_template('predict.html', prediction_text="Error: " + str(e))

    return render_template('predict.html')


# ---------------- UPLOAD ----------------
@app.route('/upload', methods=['GET','POST'])
def upload():
    if 'admin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        fig = px.scatter(df, x=df.columns[1], y=df.columns[2],
                         title="Uploaded Data Visualization")

        return render_template('upload.html',
                               graph=fig.to_html(full_html=False))

    return render_template('upload.html')


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
