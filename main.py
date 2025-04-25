# main.py
from flask import Flask, render_template, request, send_file, redirect
from scipy.stats import shapiro, skew, kurtosis, probplot, weibull_min
from scipy.special import gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
plt.rcParams['font.family'] = 'IPAexGothic'  # Prevent Japanese font from breaking
last_data = []

@app.route('/')
def index():
    return render_template("index.html", title="統計ツール | Statistical Toolkit", heading="統計ツールへようこそ / Welcome")

@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_data, message, show_plot = [], '', False
    decision_label, label_color = '', ''

    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = np.array([float(x) for x in raw_data.replace(',', ' ').split()])
            last_data = nums
            w_stat, p = shapiro(nums)
            decision_label = '正規分布といえる / Normally distributed ✅' if p > 0.05 else '正規分布ではない / Not normal ⚠'
            label_color = 'success' if p > 0.05 else 'danger'
            show_plot = True

            df = pd.DataFrame({
                'Parameter': [
                    'P値 / P-value',
                    '検定統計量（W）/ Test Statistic (W)',
                    'データ数 / Sample Size (n)',
                    '平均値 / Average (x̄)',
                    '中央値 / Median',
                    '標準偏差 / Std Deviation (S)',
                    '歪度 / Skewness',
                    '尖度 / Kurtosis'
                ],
                'Value': [
                    round(p, 8), round(w_stat, 4), len(nums),
                    round(np.mean(nums), 4), round(np.median(nums), 4),
                    round(np.std(nums, ddof=1), 5), round(skew(nums), 3), round(kurtosis(nums), 3)
                ]
            })
            table_data = df.to_dict(orient='records')
        except:
            message = "⚠ 入力形式に誤りがあります / Invalid input format"

    return render_template("shapiro.html", table_data=table_data, message=message, show_plot=show_plot,
                           decision_label=decision_label, label_color=label_color,
                           title="Shapiro-Wilk検定", heading="Shapiro-Wilk 正規性検定 / Normality Test")

@app.route('/plot.png')
def plot_png():
    global last_data
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(last_data, bins='auto', color='skyblue', edgecolor='black')
    axs[0].set_title('ヒストグラム / Histogram')
    axs[0].set_xlabel('値 / Value')
    axs[0].set_ylabel('頻度 / Frequency')
    axs[1].set_title("確率プロット / Probability Plot")
    probplot(last_data, dist="norm", plot=axs[1])
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')
