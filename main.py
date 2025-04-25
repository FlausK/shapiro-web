from flask import Flask, render_template, request, send_file
from scipy.stats import shapiro, skew, kurtosis, probplot, weibull_min
from scipy.special import gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
last_data = []


@app.route('/')
def index():
    return render_template("index.html", title="Statistical Toolkit", heading="Welcome to the Statistical Toolkit")


# --- Shapiro-Wilk 正規性検定 ---
@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_html, message, show_plot = '', '', False
    decision_label, label_color = '', ''
    plot_data = ''

    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = np.array([float(x) for x in raw_data.replace(',', ' ').split()])
            last_data = nums
            w_stat, p = shapiro(nums)
            decision_label = 'Normally distributed ✅' if p > 0.05 else 'Not normally distributed ⚠'
            label_color = 'success' if p > 0.05 else 'danger'
            show_plot = True

            # 統計表
            df = pd.DataFrame({
                'Parameter': ['P-value', 'W', 'Size', 'Average', 'Median', 'Std Dev', 'Skew', 'Kurtosis'],
                'Value': [
                    round(p, 8), round(w_stat, 4), len(nums),
                    round(np.mean(nums), 4), round(np.median(nums), 4),
                    round(np.std(nums, ddof=1), 5), round(skew(nums), 3), round(kurtosis(nums), 3)
                ]
            })
            table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)

            # 画像生成 → base64エンコードしてHTMLに埋め込み
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].hist(nums, bins='auto', color='skyblue', edgecolor='black')
            axs[0].set_title('Histogram')
            axs[1].set_title("QQ Plot")
            probplot(nums, dist="norm", plot=axs[1])
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')

        except:
            message = "⚠ Invalid input format. Please enter numbers separated by commas or spaces."

    return render_template("shapiro.html",
                           title="Shapiro-Wilk Test",
                           heading=None,
                           table_html=table_html,
                           message=message,
                           show_plot=show_plot,
                           decision_label=decision_label,
                           label_color=label_color,
                           plot_data=plot_data)



# --- Weibull 寿命分析 ---
@app.route('/lifespan', methods=['GET', 'POST'])
def lifespan_page():
    result = ''
    lifespan_input = request.form.get('lifespan_data', '').strip()
    failure_cost = request.form.get('failure_cost', '')
    maint_cost = request.form.get('maint_cost', '')
    comment = tip = ''
    cdf_path = pdf_path = ''
    cost_plot_path = ''

    if request.method == 'POST' and lifespan_input:
        try:
            failures = np.array([float(x) for x in lifespan_input.replace(',', ' ').split()])
            shape, loc, scale = weibull_min.fit(failures, floc=0)
            beta, eta = round(shape, 3), round(scale, 3)
            avg_life = round(eta * gamma(1 + 1 / beta), 2)
            result = f"Weibull Estimation: β ≒ {beta}, η ≒ {eta}, Mean ≒ {avg_life}"

            # 常に表示：点検アドバイス
            tip = f"Recommended inspection start: around {round(avg_life * 0.75, 2)} units"

            # 故障傾向コメント
            if beta < 1:
                comment = "Failure trend: Early failure (β < 1)"
            elif beta == 1:
                comment = "Failure trend: Random failure (β = 1)"
            else:
                comment = "Failure trend: Wear-out failure (β > 1)"

            # オプション：コスト最適化
            if failure_cost and maint_cost:
                fc, mc = float(failure_cost), float(maint_cost)
                min_cost, best_day = float('inf'), 0

                for t in np.linspace(max(1, eta * 0.3), eta * 3, 100):
                    prob = weibull_min.cdf(t, beta, scale=eta)
                    cost = prob * fc + mc
                    if cost < min_cost:
                        min_cost, best_day = cost, t

                # コスト比較：前後 ±10% の点
                t_minus = best_day * 0.9
                t_plus = best_day * 1.1
                c_minus = weibull_min.cdf(t_minus, beta, scale=eta) * fc + mc
                c_best = weibull_min.cdf(best_day, beta, scale=eta) * fc + mc
                c_plus = weibull_min.cdf(t_plus, beta, scale=eta) * fc + mc

                result += f"<br><b>Optimal replacement timing:</b> {round(best_day, 2)} units"
                result += f"<br>→ Cost at {round(t_minus, 2)}: {round(c_minus, 2)}"
                result += f"<br>→ Cost at {round(best_day, 2)} (best): {round(c_best, 2)}"
                result += f"<br>→ Cost at {round(t_plus, 2)}: {round(c_plus, 2)}"

                # --- コスト vs 交換タイミングのグラフ ---
                t_range = np.linspace(max(1, eta * 0.3), eta * 3, 100)
                failure_probs = weibull_min.cdf(t_range, beta, scale=eta)
                fail_costs = failure_probs * fc
                maint_costs = np.full_like(fail_costs, mc)
                total_costs = fail_costs + maint_costs

                cost_plot_path = os.path.join('static', 'cost_curve.png')
                plt.figure(figsize=(6, 4))
                plt.plot(t_range, fail_costs, label='Failure Cost', linestyle='--')
                plt.plot(t_range, maint_costs, label='Maintenance Cost', linestyle=':')
                plt.plot(t_range, total_costs, label='Total Cost', linewidth=2)
                plt.axvline(best_day, color='gray', linestyle='-', alpha=0.5, label='Optimal Timing')
                plt.xlabel("Replacement Timing")
                plt.ylabel("Cost")
                plt.title("Cost vs Replacement Timing")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(cost_plot_path)
                plt.close()


            # グラフ作成（CDF & PDF）
            t_vals = np.linspace(0, max(failures) * 1.5, 200)
            cdf = weibull_min.cdf(t_vals, beta, scale=eta)
            pdf = weibull_min.pdf(t_vals, beta, scale=eta)

            os.makedirs('static', exist_ok=True)
            cdf_path = os.path.join('static', 'weibull_cdf.png')
            pdf_path = os.path.join('static', 'weibull_pdf.png')

            plt.figure(figsize=(6, 4))
            plt.plot(t_vals, cdf)
            plt.title("Weibull CDF")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Probability")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(cdf_path)
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(t_vals, pdf)
            plt.title("Weibull PDF")
            plt.xlabel("Time")
            plt.ylabel("Density")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(pdf_path)
            plt.close()

        except:
            result = "⚠ Invalid data format. Please enter numerical values separated by commas or spaces."

    return render_template("lifespan.html", title="Lifespan Analysis", heading=None,
                       lifespan_input=lifespan_input, failure_cost=failure_cost,
                       maint_cost=maint_cost, result=result,
                       comment=comment, tip=tip,
                       cdf_path=cdf_path, pdf_path=pdf_path,
                       cost_plot_path=cost_plot_path)



# --- アプリ実行 ---
if __name__ == '__main__':
    app.run(debug=True)
