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

@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_html = ''
    message = ''
    show_plot = False
    decision_label = ''
    label_color = ''
    lifespan_result = ''
    lifespan_input = ''
    failure_cost = ''
    maint_cost = ''

    if request.method == 'POST':
        # --- Shapiro検定 ---
        raw_data = request.form.get('data', '')
        try:
            nums = [float(x) for x in raw_data.replace(',', ' ').split()]
            if len(nums) < 3:
                message = "⚠ データ数が少なすぎます。3つ以上入力してください。"
            else:
                nums = np.array(nums)
                last_data = nums
                w_stat, p = shapiro(nums)
                if p > 0.05:
                    decision_label = '正規分布といえる ✅'
                    label_color = 'success'
                else:
                    decision_label = '正規分布ではない ⚠'
                    label_color = 'danger'

                df = pd.DataFrame({
                    'Parameter': [
                        'P-value', 'W', 'Sample size (n)', 'Average (x̄)',
                        'Median', 'Sample Std Dev (S)', 'Skewness', 'Kurtosis'
                    ],
                    'Value': [
                        round(p, 8), round(w_stat, 4), len(nums),
                        round(np.mean(nums), 4), round(np.median(nums), 4),
                        round(np.std(nums, ddof=1), 5), round(skew(nums), 3), round(kurtosis(nums), 3)
                    ]
                })
                table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)
                show_plot = True
        except:
            message = "⚠ 入力形式に誤りがあります。"

        # --- 寿命分析（Weibullベース） ---
        lifespan_input = request.form.get('lifespan_data', '').strip()
        failure_cost = request.form.get('failure_cost', '').strip()
        maint_cost = request.form.get('maint_cost', '').strip()
        lifespan_result = ''

        if lifespan_input:
            try:
                failures = np.array([float(x) for x in lifespan_input.replace(',', ' ').split()])
                if len(failures) < 3:
                    lifespan_result = "⚠ データ数が少なすぎます。3つ以上入力してください。"
                else:
                    shape, loc, scale = weibull_min.fit(failures, floc=0)
                    beta = shape
                    eta = scale

                    lifespan_result = (
                        f"Weibull分布に基づく推定結果：<br>"
                        f"形状パラメータ β ≒ {round(beta, 3)}<br>"
                        f"スケールパラメータ η ≒ {round(eta, 3)}<br>"
                        f"推定される平均寿命 ≒ {round(eta * gamma(1 + 1 / beta), 2)}（単位無視）"
                    )

                    if failure_cost and maint_cost:
                        fc = float(failure_cost)
                        mc = float(maint_cost)
                        best_day = eta
                        min_cost = float('inf')

                        for t in np.linspace(eta * 0.5, eta * 2.5, 100):
                            fail_prob = weibull_min.cdf(t, beta, scale=eta)
                            expected = fail_prob * fc + mc
                            if expected < min_cost:
                                min_cost = expected
                                best_day = t

                        lifespan_result += f"<br>金額的に最適な交換タイミングは約 <b>{round(best_day, 2)}</b> です。"
            except:
                lifespan_result = "⚠ 寿命データの入力形式に誤りがあります。"

    return render_template(
        "shapiro.html",
        table_html=table_html,
        message=message,
        show_plot=show_plot,
        decision_label=decision_label,
        label_color=label_color,
        lifespan_input=lifespan_input,
        failure_cost=failure_cost,
        maint_cost=maint_cost,
        lifespan_result=lifespan_result
    )

@app.route('/plot.png')
def plot_png():
    global last_data
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(last_data, bins='auto', color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    probplot(last_data, dist="norm", plot=axs[1])
    axs[1].set_title('QQ Plot')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/')
def index():
    return redirect('/shapiro')