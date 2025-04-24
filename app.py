from flask import Flask, render_template, request
from scipy.stats import shapiro, skew, kurtosis
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = ''
    message = ''
    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = [float(x) for x in raw_data.replace(',', ' ').split()]
            if len(nums) < 3:
                message = "⚠ データ数が少なすぎます。3つ以上入力してください。"
            else:
                nums = np.array(nums)
                w_stat, p = shapiro(nums)
                mean = np.mean(nums)
                median = np.median(nums)
                std = np.std(nums, ddof=1)
                skewness = skew(nums)
                kurt = kurtosis(nums)
                n = len(nums)

                df = pd.DataFrame({
                    'Parameter': [
                        'P-value',
                        'W',
                        'Sample size (n)',
                        'Average (x̄)',
                        'Median',
                        'Sample Std Dev (S)',
                        'Skewness',
                        'Kurtosis'
                    ],
                    'Value': [
                        round(p, 8),
                        round(w_stat, 4),
                        n,
                        round(mean, 4),
                        round(median, 4),
                        round(std, 5),
                        round(skewness, 3),
                        round(kurt, 3)
                    ]
                })

                # HTMLテーブルへ変換（クラスを付けておく）
                table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)
        except:
            message = "⚠ 入力形式に誤りがあります。数値をスペースまたはカンマで区切ってください。"

    return render_template("index.html", table_html=table_html, message=message)
