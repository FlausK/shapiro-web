from flask import Flask, render_template, request
import pandas as pd
from scipy.stats import shapiro

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            col = df.select_dtypes(include='number').columns[0]
            data = df[col].dropna()
            stat, p = shapiro(data)
            result = f'Shapiro-Wilk検定: p値 = {p:.4f} → {"正規分布" if p > 0.05 else "正規分布ではない"}'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
