from flask import Flask, render_template, request, send_file, redirect
from scipy.stats import shapiro, skew, kurtosis, probplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # キャッシュ防止

# 保存するデータをグローバル変数で保持
last_data = []

@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_html = ''
    message = ''
    show_plot = False
    decision_label = ''
    label_color = ''


    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = [float(x) for x in raw_data.replace(',', ' ').split()]
            if len(nums) < 3:
                message = "⚠ データ数が少なすぎます。3つ以上入力してください。"
            else:
                nums = np.array(nums)
                last_data = nums  # プロット用に保存
                w_stat, p = shapiro(nums)

                decision_label = ''
                label_color = ''
                if p > 0.05:
                    decision_label = '正規分布といえる ✅'
                    label_color = 'success'
                else:
                    decision_label = '正規分布ではない ⚠'
                    label_color = 'danger'

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

                table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)
                show_plot = True
        except:
            message = "⚠ 入力形式に誤りがあります。数値をスペースまたはカンマで区切ってください。"

    return render_template(
        "shapiro.html",
        table_html=table_html,
        message=message,
        show_plot=show_plot,
        decision_label=decision_label,
        label_color=label_color
    )


@app.route('/plot.png')
def plot_png():
    global last_data
    import matplotlib.pyplot as plt
    from scipy.stats import probplot
    import io

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # ヒストグラム
    axs[0].hist(last_data, bins='auto', color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    # QQプロット
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
