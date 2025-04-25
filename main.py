from flask import Flask, render_template, request, send_file, redirect
from scipy.stats import shapiro, skew, kurtosis, probplot, weibull_min
from scipy.special import gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
last_data = []

@app.route('/')
def index():
    return render_template("index.html", title="統計ツール", heading="統計ツールへようこそ")

# --- Shapiro-Wilk 正規性検定 ---
@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_html, message, show_plot = '', '', False
    decision_label, label_color = '', ''

    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = np.array([float(x) for x in raw_data.replace(',', ' ').split()])
            last_data = nums
            w_stat, p = shapiro(nums)
            decision_label = '正規分布といえる ✅' if p > 0.05 else '正規分布ではない ⚠'
            label_color = 'success' if p > 0.05 else 'danger'
            show_plot = True

            df = pd.DataFrame({
                'Parameter': ['P-value', 'W', 'Size', 'Average', 'Median', 'Std Dev', 'Skew', 'Kurtosis'],
                'Value': [
                    round(p, 8), round(w_stat, 4), len(nums),
                    round(np.mean(nums), 4), round(np.median(nums), 4),
                    round(np.std(nums, ddof=1), 5), round(skew(nums), 3), round(kurtosis(nums), 3)
                ]
            })
            table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)
        except:
            message = "⚠ 入力形式に誤りがあります"

    return render_template("shapiro.html", title="Shapiro-Wilk検定", heading="Shapiro-Wilk 正規性検定",
                           table_html=table_html, message=message, show_plot=show_plot,
                           decision_label=decision_label, label_color=label_color)

# --- 寿命分析 ---
@app.route('/lifespan', methods=['GET', 'POST'])
def lifespan_page():
    result = ''
    lifespan_input = request.form.get('lifespan_data', '').strip()
    failure_cost = request.form.get('failure_cost', '')
    maint_cost = request.form.get('maint_cost', '')

    if request.method == 'POST' and lifespan_input:
        try:
            failures = np.array([float(x) for x in lifespan_input.replace(',', ' ').split()])
            shape, loc, scale = weibull_min.fit(failures, floc=0)
            beta, eta = shape, scale
            avg_life = eta * gamma(1 + 1 / beta)

            result = f"Weibull推定 β ≒ {round(beta, 3)}, η ≒ {round(eta, 3)}, 平均寿命 ≒ {round(avg_life, 2)}"
            if failure_cost and maint_cost:
                fc, mc = float(failure_cost), float(maint_cost)
                min_cost = float('inf')
                for t in np.linspace(eta * 0.5, eta * 2.5, 100):
                    prob = weibull_min.cdf(t, beta, scale=eta)
                    cost = prob * fc + mc
                    if cost < min_cost:
                        min_cost, best_day = cost, t
                result += f"<br>最適交換時期：<b>{round(best_day, 2)}</b> 単位"
        except:
            result = "⚠ データ形式が正しくありません"

    return render_template("lifespan.html", title="寿命分析", heading="寿命分析（Weibull分布）",
                           lifespan_input=lifespan_input, failure_cost=failure_cost,
                           maint_cost=maint_cost, result=result)

# --- グラフ用 ---
@app.route('/plot.png')
def plot_png():
    global last_data
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(last_data, bins='auto', color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram')
    axs[1].set_title("QQ Plot")
    probplot(last_data, dist="norm", plot=axs[1])
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')
